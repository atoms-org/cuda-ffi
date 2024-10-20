from __future__ import annotations

import ast
import inspect
import textwrap
from enum import Enum, auto
from functools import update_wrapper
from typing import Any, Callable, ParamSpec, TypeVar, cast, overload

from .args import CudaArg, CudaArgDirection, CudaDataType
from .device import init
from .graph.graph import CudaGraph, GraphNode
from .graph.kernel import CudaKernelNode
from .graph.memcpy import CudaMemcpyNode
from .memory import HostBuffer
from .module import CudaFunction, CudaFunctionNameNotFound, CudaModule

ModType = dict[str, str | CudaModule]
AnyFn = Callable[[Any], Any]

P = ParamSpec("P")
R = TypeVar("R")


# adapted from https://lemonfold.io/posts/2022/dbc/typed_decorator/
@overload
def cuda_plan(func: Callable[P, R]) -> Callable[P, R]: ...


@overload
def cuda_plan(
    *,
    modules: dict[str, CudaModule] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def cuda_plan(
    func: Callable[P, R] | None = None,
    *,
    modules: dict[str, CudaModule] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    # Without arguments `func` is passed directly to the decorator
    if func is not None:
        if not callable(func):
            raise TypeError("CudaPlan is not a callable. Did you use a non-function argument?")
        return cast(Callable[P, R], CudaPlan(func, modules=modules))

    # With arguments, we need to return a function that accepts the function
    def _decorator(func: Callable[P, R]) -> Callable[P, R]:
        return cast(Callable[P, R], CudaPlan(func, modules=modules))

    return _decorator


class CudaPlanException(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class CudaModuleNotFound(Exception):
    pass


class CudaPlan:
    def __init__(
        self,
        fn: Callable[..., Any],
        modules: dict[str, CudaModule] | None = None,
    ) -> None:
        init()

        self.src_text = textwrap.dedent(inspect.getsource(fn))
        self.src_ast = ast.parse(self.src_text)
        self.vars: list[CudaPlanVar] = []
        self.modules = modules

        update_wrapper(self, fn)

        # source should be a single function
        if len(self.src_ast.body) != 1:
            raise CudaPlanException(
                f"cuda plan expected exactly one function, got {len(self.src_ast.body)}"
            )

        if not isinstance(self.src_ast.body[0], ast.FunctionDef):
            raise CudaPlanException("expected body of cuda plan to be a single function")

        # import pprint

        # pprint.pprint(ast.dump(self.src_ast))

        # get args from function definition
        self.fn_def_ast = self.src_ast.body[0]
        self.fn_def_args_ast = self.fn_def_ast.args.args
        self.inputs: list[CudaPlanVar] = []
        for n in range(len(self.fn_def_args_ast)):
            arg = self.fn_def_args_ast[n]
            arg_name = arg.arg
            if arg.annotation is not None:
                assert isinstance(arg.annotation, ast.Name)
                annote = arg.annotation.id
            else:
                annote = "Any"
            self.inputs.append(self.resolve_var(arg_name, CudaPlanVarType.arg))

        # decode overall return type
        if self.fn_def_ast.returns is not None:
            match self.fn_def_ast.returns:
                case ast.Name():
                    self.output_type = self.fn_def_ast.returns.id
                case ast.Constant():
                    self.output_type = str(self.fn_def_ast.returns.value)
                case ast.Subscript():
                    self.output_type = ast.unparse(self.fn_def_ast.returns)
                case _:
                    raise CudaPlanException(
                        f"unknown return type: '{self.fn_def_ast.returns.__class__.__name__}'"
                    )
        else:
            self.output_type = "Any"

        # decode statements
        self.steps: list[CudaPlanStep] = []
        no_more = False
        for stmt in self.fn_def_ast.body:
            if no_more:
                raise CudaPlanException("statements found after return in CudaPlan")

            step = CudaPlanStep(self, stmt)
            self.steps.append(step)

            if step.is_return:
                no_more = True

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        g = self.to_graph(*args)
        g.run()

    def resolve_var(self, name: str, type: CudaPlanVarType, val: Any = None) -> CudaPlanVar:
        # always make new constants
        if type == CudaPlanVarType.constant:
            v = CudaPlanVar(name, type, val)
            self.vars.append(v)
            return v

        # if it's a real variable, return the existing variable
        for var in self.vars:
            if var.name == name:
                return var

        # if no variable is found, make a new one
        v = CudaPlanVar(name, type)
        self.vars.append(v)
        return v

    def to_graph(self, *args: Any) -> CudaGraph:
        g = CudaGraph()

        # clear any old var values
        for var in self.vars:
            if var.type == CudaPlanVarType.arg:
                var.reset()
            if var.type == CudaPlanVarType.constant and var.inferred_type is None:
                var.inferred_type = CudaDataType.resolve(var.val)

        # assign input args
        if len(self.inputs) != len(args):
            raise CudaPlanException(
                f"wrong number of arguments to CudaPlan: expected {len(self.inputs)}, got {len(args)}"
            )

        for input, arg in zip(self.inputs, args):
            ca = CudaArg(arg)
            input.inferred_type = ca.data_type
            input.val = ca.nv_data
            if ca.is_pointer:
                assert ca.byte_size is not None
                assert ca.dev_mem is not None
                buf = ca.data_type.encode(ca.data)
                assert not isinstance(buf, float | int)
                input.host_buf = HostBuffer(buf)
                input.input_arg = ca
                input.last_use = CudaMemcpyNode(g, input.host_buf, ca.dev_mem, ca.byte_size)

        # create nodes to copy inputs to device
        # for input in self.inputs:
        # CudaArgList(args, fn.arg_types)
        # start_nodes = arg_list.create_copy_to_device_nodes(g)

        # create nodes for each step in the plan
        for step in self.steps:
            if step.call_fn is not None:
                # create call args
                for v in step.input_vars:
                    if v.val is None:
                        raise Exception(f"can't use unassigned cuda plan variable '{v.name}'")
                    if v.type == CudaPlanVarType.constant and v.inferred_type is None:
                        v.inferred_type = CudaDataType.resolve(v.val)

                arg_data = [arg.val for arg in step.input_vars]
                arg_types = [
                    arg.inferred_type.get_ctype(arg.val)
                    for arg in step.input_vars
                    if arg.inferred_type is not None
                ]
                assert len(arg_data) == len(arg_types)
                arg_list = (tuple(arg_data), tuple(arg_types))

                # TODO: check to make sure we have the same number of args and
                # the same ctypes

                # create kernel node
                kn = CudaKernelNode(g, step.call_fn, arg_list)

                # create memcpys, create dependencies, save var state
                assert step.call_fn.arg_types is not None  # TODO
                for var, arg_type in zip(step.input_vars, step.call_fn.arg_types):
                    if CudaArgDirection.is_input(arg_type.direction):
                        if var.last_use is not None:
                            kn.depends_on(var.last_use)
                        var.last_use = kn
                    if var.type == CudaPlanVarType.arg:
                        var.last_direction = arg_type.direction

        # do output args
        for input in self.inputs:
            if input.last_direction is not None and CudaArgDirection.is_output(
                input.last_direction
            ):
                assert input.host_buf is not None
                assert input.input_arg is not None
                assert input.input_arg.dev_mem is not None
                assert input.input_arg.byte_size is not None
                mcn = CudaMemcpyNode(
                    g, input.input_arg.dev_mem, input.host_buf, input.input_arg.byte_size
                )
                if input.last_use is not None:
                    mcn.depends_on(input.last_use)
                input.last_use = mcn

        return g


class CudaPlanVarType(Enum):
    constant = auto()
    arg = auto()


class CudaPlanVar:
    def __init__(self, name: str, type: CudaPlanVarType, val: Any = None):
        self.name = name
        self.type = type
        self.val = val
        self.last_use: GraphNode | None = None
        self.last_direction: CudaArgDirection | None = None
        self.inferred_type: CudaDataType[Any] | None = None
        self.host_buf: HostBuffer | None = None
        self.input_arg: CudaArg | None = None

    def reset(self) -> None:
        self.val = None
        self.last_use = None
        self.last_type = None

    def __str__(self) -> str:
        return f"CudaPlanVar(name='{self.name}', type='{self.type.name}', val='{self.val}')"


class CudaPlanStep:
    def __init__(self, plan: CudaPlan, stmt: ast.stmt) -> None:
        self.plan = plan
        self.exec: CudaFunction | CudaPlan | None = None
        self.input_vars: list[CudaPlanVar] = []
        self.input_kwvars: dict[str, CudaPlanVar] = {}
        self.output_vars: list[CudaPlanVar] = []
        self.is_return: bool = False
        self.call_fn: CudaFunction | None = None

        match stmt:
            case ast.Assign():
                self.output_vars = self._decode_assignment_targets(stmt)
                assert isinstance(stmt.value, ast.Call)
                self.call_fn = self._decode_call_function(stmt.value)
                self.input_vars, self.input_kwvars = self._decode_call_args(stmt.value)
            case ast.Return():
                self.is_return = True
                match stmt.value:
                    case ast.Name():
                        self.output_vars.append(
                            self.plan.resolve_var(stmt.value.id, CudaPlanVarType.arg)
                        )
                        self.input_vars.append(
                            self.plan.resolve_var(stmt.value.id, CudaPlanVarType.arg)
                        )
                    case ast.Tuple():
                        tup: ast.Tuple = stmt.value
                        for e in tup.elts:
                            assert isinstance(e, ast.Name)
                            self.output_vars.append(
                                self.plan.resolve_var(e.id, CudaPlanVarType.arg)
                            )
                            self.input_vars.append(self.plan.resolve_var(e.id, CudaPlanVarType.arg))
                    case ast.Call():
                        self.call_fn = self._decode_call_function(stmt.value)
                        self.input_vars, self.input_kwvars = self._decode_call_args(stmt.value)
                    case ast.Constant():
                        self.output_vars.append(
                            self.plan.resolve_var(
                                str(stmt.value.value),
                                CudaPlanVarType.constant,
                                val=stmt.value.value,
                            )
                        )
                    case _:
                        raise CudaPlanException(
                            f"unknown return value: '{stmt.value.__class__.__name__}'"
                        )
            case ast.Expr():
                if not isinstance(stmt.value, ast.Call):
                    raise CudaPlanException(
                        f"unknown expression type: '{stmt.value.__class__.__name__}'"
                    )
                self.call_fn = self._decode_call_function(stmt.value)
                self.input_vars, self.input_kwvars = self._decode_call_args(stmt.value)
            case _:
                raise CudaPlanException(f"unknown statement type: '{stmt.__class__.__name__}'")

    def _decode_assignment_targets(self, assn: ast.Assign) -> list[CudaPlanVar]:
        ret: list[CudaPlanVar] = []

        for tgt in assn.targets:
            match tgt:
                case ast.Name():
                    ret.append(self.plan.resolve_var(tgt.id, CudaPlanVarType.arg))
                case ast.Tuple():
                    tup: ast.Tuple = tgt
                    for e in tup.elts:
                        assert isinstance(e, ast.Name)
                        ret.append(self.plan.resolve_var(e.id, CudaPlanVarType.arg))
                case _:
                    raise CudaPlanException(
                        f"unknown assignment target: '{tgt.__class__.__name__}'"
                    )

        return ret

    def _decode_call_function(self, call: ast.Call) -> CudaFunction:
        ret_fn: CudaFunction

        match call.func:
            case ast.Name():
                fn_name = call.func.id
                fn = CudaModule.find_function(fn_name)
                if fn is None:
                    raise CudaFunctionNameNotFound(
                        f"function named '{fn_name}' in CudaPlan not found"
                    )
                ret_fn = fn
            case ast.Attribute():
                assert isinstance(call.func.value, ast.Name)
                mod_name = call.func.value.id
                fn_name = call.func.attr
                if self.plan.modules is None:
                    raise CudaModuleNotFound(
                        f"trying to find module named '{mod_name}' but no modules were specified for CudaPlan"
                    )
                if not mod_name in self.plan.modules:
                    raise CudaModuleNotFound(f"module '{mod_name}' not found in CudaPlan")
                mod = self.plan.modules[mod_name]
                ret_fn = mod.get_function(fn_name)
            case _:
                raise CudaPlanException(f"unknown call name: '{call.func.__class__.__name__}'")

        if ret_fn.arg_types is None:
            raise CudaPlanException(
                f"function '{ret_fn.name}' didn't have any arg types and arg types are required for CudaPlan"
            )
        return ret_fn

    def _decode_call_args(self, call: ast.Call) -> tuple[list[CudaPlanVar], dict[str, CudaPlanVar]]:
        args_ret: list[CudaPlanVar] = []
        kwargs_ret: dict[str, CudaPlanVar] = {}

        for arg in call.args:
            match arg:
                case ast.Name():
                    args_ret.append(self.plan.resolve_var(arg.id, CudaPlanVarType.arg))
                case ast.Constant():
                    args_ret.append(
                        self.plan.resolve_var(
                            str(arg.value), CudaPlanVarType.constant, val=arg.value
                        )
                    )
                case _:
                    raise CudaPlanException(f"unknown call argument: '{arg.__class__.__name__}'")

        for kwarg in call.keywords:
            match arg:
                case ast.Name():
                    key = kwarg.arg
                    if not key in ["block", "grid"]:
                        raise CudaPlanException(
                            "only 'block' and 'grid' keyword args are currently allowed"
                        )
                    match kwarg.value:
                        case ast.Tuple():
                            if len(kwarg.value.elts) != 3:
                                raise CudaPlanException(
                                    "only tuple of three integers is currently allowed for kwargs"
                                )

                            tuple_vals: list[int] = []
                            for e in kwarg.value.elts:
                                if not isinstance(e, ast.Constant) or not isinstance(e.value, int):
                                    raise CudaPlanException(
                                        "only tuple of three integers is currently allowed for kwargs"
                                    )
                                tuple_vals.append(e.value)

                            val_str = ast.unparse(kwarg.value)
                            kwargs_ret[key] = CudaPlanVar(
                                val_str, CudaPlanVarType.constant, tuple(tuple_vals)
                            )
                            print("kwarg.value", kwarg.value)
                        case _:
                            raise CudaPlanException(
                                "only tuple of three integers is currently allowed for kwargs"
                            )
                case ast.Constant():
                    key = kwarg.arg
                    assert key is not None
                    kwargs_ret[key] = CudaPlanVar(
                        str(kwarg.value), CudaPlanVarType.constant, val=kwarg.value
                    )
                case _:
                    raise CudaPlanException(
                        f"unknown call keyword argument: '{arg.__class__.__name__}'"
                    )

        return args_ret, kwargs_ret
