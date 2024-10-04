import ast
import inspect
import textwrap
from enum import Enum, auto
from functools import update_wrapper
from typing import Any, Callable, ParamSpec, TypeVar, cast

from cudaffi.module import CudaFunction, CudaModule

from .graph.graph import CudaGraph

ModType = dict[str, str | CudaModule]
AnyFn = Callable[[Any], Any]

P = ParamSpec("P")
R = TypeVar("R")

# def cuda_plan() -> Callable[..., Any]:
#     def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
#         @wraps(fn)
#         def wrapper() -> CudaPlan:
#             return CudaPlan(fn)

#         return wrapper

#     return decorator


def cuda_plan(fn: Callable[P, R]) -> Callable[P, R]:
    return cast(Callable[P, R], CudaPlan(fn))


class CudaPlanException(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class CudaPlan:
    def __init__(self, fn: Callable[..., Any]) -> None:
        self.src_text = textwrap.dedent(inspect.getsource(fn))
        self.src_ast = ast.parse(self.src_text)

        update_wrapper(self, fn)

        # source should be a single function
        if len(self.src_ast.body) != 1:
            raise CudaPlanException(
                f"cuda plan expected exactly one function, got {len(self.src_ast.body)}"
            )

        if not isinstance(self.src_ast.body[0], ast.FunctionDef):
            raise CudaPlanException("expected body of cuda plan to be a single function")

        import pprint

        pprint.pprint(ast.dump(self.src_ast))

        # TODO: get args from function definition
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
            # TODO: new CudaPlanVar

        if self.fn_def_ast.returns is not None:
            match self.fn_def_ast.returns:
                case ast.Name():
                    self.output_type = self.fn_def_ast.returns.id
                case ast.Constant():
                    self.output_type = str(self.fn_def_ast.returns.value)
                    print("output type constant:", self.output_type)
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

            step = CudaPlanStep(stmt)
            self.steps.append(step)

            if step.is_return:
                no_more = True

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        print("calling CudaPlan")

    def to_graph(self) -> CudaGraph:
        g = CudaGraph()
        # TODO: for each statement...
        return g


class CudaPlanVarType(Enum):
    constant = auto()
    arg = auto()
    kwarg = auto()


class CudaPlanVar:
    def __init__(self, name: str, type: CudaPlanVarType, val: Any = None):
        self.name = name
        self.type = type
        self.val = val


class CudaPlanStep:
    def __init__(self, stmt: ast.stmt) -> None:
        self.exec: CudaFunction | CudaPlan | None = None
        self.input_vars: list[CudaPlanVar] = []
        self.input_kwvars: dict[str, CudaPlanVar] = {}
        self.output_vars: list[CudaPlanVar] = []
        self.is_return: bool = False
        self.call_name: str | None = None
        self.call_module: str | None = None

        match stmt:
            case ast.Assign():
                self.output_vars = _decode_assignment_targets(stmt)
                assert isinstance(stmt.value, ast.Call)
                self.call_name, self.call_module = _decode_call_name(stmt.value)
                self.input_vars, self.input_kwvars = _decode_call_args(stmt.value)
            case ast.Return():
                self.is_return = True
                match stmt.value:
                    case ast.Name():
                        self.output_vars.append(CudaPlanVar(stmt.value.id, CudaPlanVarType.arg))
                        self.input_vars.append(CudaPlanVar(stmt.value.id, CudaPlanVarType.arg))
                    case ast.Tuple():
                        tup: ast.Tuple = stmt.value
                        for e in tup.elts:
                            assert isinstance(e, ast.Name)
                            self.output_vars.append(CudaPlanVar(e.id, CudaPlanVarType.arg))
                            self.input_vars.append(CudaPlanVar(e.id, CudaPlanVarType.arg))
                    case ast.Call():
                        self.call_name, self.call_module = _decode_call_name(stmt.value)
                        self.input_vars, self.input_kwvars = _decode_call_args(stmt.value)
                    case ast.Constant():
                        self.output_vars.append(
                            CudaPlanVar(
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
                self.call_name, self.call_module = _decode_call_name(stmt.value)
                self.input_vars, self.input_kwvars = _decode_call_args(stmt.value)
            case _:
                raise CudaPlanException(f"unknown statement type: '{stmt.__class__.__name__}'")


def _decode_assignment_targets(assn: ast.Assign) -> list[CudaPlanVar]:
    ret: list[CudaPlanVar] = []

    for tgt in assn.targets:
        match tgt:
            case ast.Name():
                ret.append(CudaPlanVar(tgt.id, CudaPlanVarType.arg))
            case ast.Tuple():
                tup: ast.Tuple = tgt
                for e in tup.elts:
                    assert isinstance(e, ast.Name)
                    ret.append(CudaPlanVar(e.id, CudaPlanVarType.arg))
            case _:
                raise CudaPlanException(f"unknown assignment target: '{tgt.__class__.__name__}'")

    return ret


def _decode_call_name(call: ast.Call) -> tuple[str, str | None]:
    match call.func:
        case ast.Name():
            return (call.func.id, None)
        case ast.Attribute():
            assert isinstance(call.func.value, ast.Name)
            mod_name = call.func.value.id
            fn_name = call.func.attr
            return (fn_name, mod_name)
        case _:
            raise CudaPlanException(f"unknown call name: '{call.func.__class__.__name__}'")


def _decode_call_args(call: ast.Call) -> tuple[list[CudaPlanVar], dict[str, CudaPlanVar]]:
    args_ret: list[CudaPlanVar] = []
    kwargs_ret: dict[str, CudaPlanVar] = {}

    for arg in call.args:
        match arg:
            case ast.Name():
                args_ret.append(CudaPlanVar(arg.id, CudaPlanVarType.arg))
            case ast.Constant():
                args_ret.append(
                    CudaPlanVar(str(arg.value), CudaPlanVarType.constant, val=arg.value)
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
                        val_str, CudaPlanVarType.constant, val=eval(val_str)
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
