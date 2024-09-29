import ast
import inspect
import textwrap
import time
from functools import partial, wraps
from typing import Any, Callable

from cudaffi.module import CudaFunction, CudaModule

ModType = dict[str, str | CudaModule]
AnyFn = Callable[[Any], Any]


def cuda_plan(
    func: AnyFn | None = None, *, seconds: int | None = None, msg: str | None = None
) -> AnyFn:
    if func is None:
        return partial(cuda_plan, seconds=seconds, msg=msg)

    seconds = seconds if seconds else 1
    msg = msg if msg else "Sleeping for {} seconds".format(seconds)

    @wraps(func)
    def wrapper(*args: list[Any], **kwargs: dict[str, Any]) -> Any:
        print(msg)
        print("before")
        time.sleep(seconds)
        ret = func(*args, **kwargs)
        print("after")
        return ret

    return wrapper


class CudaPlanException(Exception):
    pass


class CudaPlan:
    def __init__(self, fn: Callable[..., Any]) -> None:
        self.src_text = textwrap.dedent(inspect.getsource(fn))
        self.src_ast = ast.parse(self.src_text)

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
            assert isinstance(self.fn_def_ast.returns, ast.Name)
            self.output_type = self.fn_def_ast.returns.id
        else:
            self.output_type = "Any"

        # decode statements
        self.steps: list[CudaPlanStep] = []
        no_more = False
        for stmt in self.fn_def_ast.body:
            if no_more:
                raise Exception("statements found after return in cuda plan")

            step = CudaPlanStep(stmt)
            self.steps.append(step)

            if step.is_return:
                no_more = True


class CudaPlanVar:
    def __init__(self, name: str, type: str | None = None):
        self.name = name


class CudaPlanStep:
    def __init__(self, stmt: ast.stmt) -> None:
        self.exec: CudaFunction | CudaPlan | None = None
        self.input_vars: list[CudaPlanVar] = []
        self.output_vars: list[CudaPlanVar] = []
        self.is_return: bool = False
        self.call_name: str | None = None
        self.call_module: str | None = None
        self.input_arg_names: list[str] = []
        self.output_arg_names: list[str] = []

        match stmt:
            case ast.Assign():
                print("statement type: assignment")
                self.output_arg_names = _decode_assignment_targets(stmt)
                assert isinstance(stmt.value, ast.Call)
                self.call_name, self.call_module = _decode_call_name(stmt.value)
                self.input_arg_names = _decode_call_args(stmt.value)
            case ast.Return():
                self.is_return = True
                match stmt.value:
                    case ast.Name():
                        self.output_arg_names.append(stmt.value.id)
                        self.input_arg_names.append(stmt.value.id)
                    case ast.Tuple():
                        tup: ast.Tuple = stmt.value
                        for e in tup.elts:
                            assert isinstance(e, ast.Name)
                            self.output_arg_names.append(e.id)
                            self.input_arg_names.append(e.id)
                    case ast.Call():
                        self.call_name, self.call_module = _decode_call_name(stmt.value)
                        self.input_arg_names = _decode_call_args(stmt.value)
                    case _:
                        raise Exception(f"unknown return value: '{stmt.value.__class__.__name__}'")

                print("statement type: return")
                # decode_return(stmt)
            case ast.Expr():
                print("statement type: expression")
                print("is call", isinstance(stmt.value, ast.Call))
                assert isinstance(stmt.value, ast.Call)
                self.call_name, self.call_module = _decode_call_name(stmt.value)
                self.input_arg_names = _decode_call_args(stmt.value)
            case _:
                raise Exception(f"unknown statement type: '{stmt.__class__.__name__}'")


def _decode_assignment_targets(assn: ast.Assign) -> list[str]:
    ret: list[str] = []

    for tgt in assn.targets:
        match tgt:
            case ast.Name():
                ret.append(tgt.id)
            case ast.Tuple():
                tup: ast.Tuple = tgt
                for e in tup.elts:
                    assert isinstance(e, ast.Name)
                    ret.append(e.id)
            case _:
                raise Exception(f"unknown assignment target: '{tgt.__class__.__name__}'")

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
            raise Exception(f"unknown call name: '{call.func.__class__.__name__}'")


def _decode_call_args(call: ast.Call) -> list[str]:
    ret: list[str] = []

    for arg in call.args:
        match arg:
            case ast.Name():
                ret.append(arg.id)
            case _:
                raise Exception(f"unknown call argument: '{arg.__class__.__name__}'")

    return ret
