from __future__ import annotations

import ast
import inspect
import textwrap
from enum import Enum, auto
from functools import update_wrapper
from typing import Any, Callable, ParamSpec, TypeVar, cast, overload

from cudaffi.module import CudaFunction, CudaFunctionNameNotFound, CudaModule

from .graph.graph import CudaGraph

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
            raise TypeError("Not a callable. Did you use a non-keyword argument?")
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
        self.inputs: list[tuple[str, str]] = []
        for n in range(len(self.fn_def_args_ast)):
            arg = self.fn_def_args_ast[n]
            arg_name = arg.arg
            if arg.annotation is not None:
                assert isinstance(arg.annotation, ast.Name)
                annote = arg.annotation.id
            else:
                annote = "Any"
            self.inputs.append((arg_name, annote))
            self.resolve_var(arg_name, CudaPlanVarType.arg)
            # TODO: new CudaPlanVar

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
        print("calling CudaPlan")

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

    def to_graph(self) -> CudaGraph:
        g = CudaGraph()
        for step in self.steps:
            print("step")
        return g


class CudaPlanVarType(Enum):
    constant = auto()
    arg = auto()


class CudaPlanVar:
    def __init__(self, name: str, type: CudaPlanVarType, val: Any = None):
        self.name = name
        self.type = type
        self.val = val


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
        match call.func:
            case ast.Name():
                fn_name = call.func.id
                fn = CudaModule.find_function(fn_name)
                if fn is None:
                    raise CudaFunctionNameNotFound(
                        f"function named '{fn_name}' in CudaPlan not found"
                    )
                return fn
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
                return mod.get_function(fn_name)
            case _:
                raise CudaPlanException(f"unknown call name: '{call.func.__class__.__name__}'")

        # TODO: resolve function from specified module

    def _decode_call_args(self, call: ast.Call) -> tuple[list[CudaPlanVar], dict[str, CudaPlanVar]]:
        args_ret: list[CudaPlanVar] = []
        kwargs_ret: dict[str, CudaPlanVar] = {}

        for arg in call.args:
            match arg:
                case ast.Name():
                    args_ret.append(CudaPlanVar(arg.id, CudaPlanVarType.arg))
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
                    assert key is not None
                    if isinstance(kwarg.value, ast.Name):
                        kwargs_ret[key] = CudaPlanVar(kwarg.value.id, CudaPlanVarType.arg)
                    else:
                        val_str = ast.unparse(kwarg.value)
                        kwargs_ret[key] = CudaPlanVar(
                            val_str,
                            CudaPlanVarType.constant,
                            val=eval(val_str),  # TODO: remove eval
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
