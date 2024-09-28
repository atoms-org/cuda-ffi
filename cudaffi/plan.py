import ast
import inspect
import textwrap
import time
from functools import partial, wraps
from typing import Any, Callable, cast

from cudaffi.module import CudaModule

ModType = dict[str, str | CudaModule]
AnyFn = Callable[[Any], Any]


# class cuda_plan:
#     @overload
#     def __init__(self, fn: AnyFn) -> None: ...

#     @overload
#     def __init__(self, *, modules: ModType | None = None) -> None: ...

#     def __init__(self, fn: AnyFn | None = None, *, modules: ModType | None = None) -> None:
#         # TODO: convert strings to modules with the specified name
#         self.modules = modules
#         self.fn = fn

#     def __call__(self, *args: list[Any]) -> Any:
#         if self.fn is None:
#             return partial(cuda_plan, modules=self.modules)

#         wraps(self.fn)(self)

#         def wrapped_f(*args: list[Any]) -> int:
#             # !!! before decorator
#             print("before")
#             ret = self.fn(*args)
#             # !!! after decorator
#             print("after")
#             return 42

#         return wrapped_f


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


class CudaPlanVar:
    pass


class CudaPlanStmt:
    pass


def parse_plan(fn: Any) -> None:
    assert callable(fn)
    fn_source_code = textwrap.dedent(inspect.getsource(fn))
    print("fn_source_code", fn_source_code)
    tree = ast.parse(fn_source_code)
    print("tree", tree)

    import pprint

    pprint.pprint(ast.dump(tree))

    assert len(tree.body) == 1

    # function definition
    assert isinstance(tree.body[0], ast.FunctionDef)
    fn_def = tree.body[0]
    print("fn_def.name", fn_def.name)
    print("fn_def.args", fn_def.args)
    fn_def_args = fn_def.args.args
    # function args definition
    assert len(fn_def_args) == 2
    for n in range(len(fn_def_args)):
        arg = fn_def_args[n]
        annote = decode_expr(arg.annotation) if arg.annotation else "None"
        print(f"function def arg {arg.arg}, type {annote}")
    # # function arg1 definition
    # assert fn_def_args[0].arg == "arg1"
    # def_arg1 = fn_def_args[0]
    # assert isinstance(def_arg1.annotation, ast.Name)
    # assert def_arg1.annotation.id == "int"
    # # function arg2 definition
    # assert fn_def_args[1].arg == "arg2"
    # assert isinstance(fn_def_args[1].annotation, ast.Name)
    # assert fn_def_args[1].annotation.id == "str"
    # function body definition
    fn_def_body = fn_def.body
    # assert isinstance(fn_def_body, list)
    # assert len(fn_def_body) == 3
    # statements
    print("fn_def_body[0]", fn_def_body[0])
    for stmt in fn_def_body:
        decode_statement(stmt)


def decode_name(n: ast.Name) -> str:
    return n.id


def decode_expr(expr: ast.expr) -> str | list[str]:
    match expr:
        case ast.Name():
            return decode_name(expr)
        case ast.Tuple():
            return decode_tuple(expr)
        case ast.Call():
            return decode_call(expr)
        case ast.Constant():
            return decode_constant(expr)
        case ast.Return():
            return decode_return(expr)
        case ast.Attribute():
            return decode_attribute(expr)
        case _:
            print("fields", expr._fields)
            raise Exception(
                f"unknown expression type: '{expr.__class__.__name__}' {expr.lineno}:{expr.col_offset}"
            )


def decode_attribute(attr: ast.Attribute) -> str | list[str]:
    print(f"attribute '{decode_expr(attr.value)}.{attr.attr}'")
    return f"{decode_expr(attr.value)}.{attr.attr}"


def decode_statement(stmt: ast.stmt) -> None:
    match stmt:
        case ast.Assign():
            print("assignment")
            decode_assignment(stmt)
        case ast.Return():
            print("return")
            decode_return(stmt)
        case _:
            raise Exception(f"unknown statement type: '{stmt.__class__.__name__}'")


def decode_return(r: ast.Return) -> str | list[str]:
    print("return value", r.value)
    val = decode_expr(r.value) if r.value is not None else "None"
    print("return value", val)
    return val


def decode_constant(c: ast.Constant) -> str | list[str]:
    print("constant", c)
    return cast(str, c.value)


def decode_call(c: ast.Call) -> str | list[str]:
    print("function name:", decode_expr(c.func))
    for n in range(len(c.args)):
        print(f"arg{n}: {decode_expr(c.args[n])}")
    return decode_expr(c.func)


def decode_tuple(tup: ast.Tuple) -> list[str]:
    ret: list[str] = []
    for e in tup.elts:
        s = decode_expr(e)
        assert not isinstance(s, list)
        ret.append(s)
    return ret


# decode_statement
def decode_assignment(ass: ast.Assign) -> None:
    print(ass.targets)
    for t in ass.targets:
        print("assignment target:", decode_expr(t))
    print("value", decode_expr(ass.value))


# decode_return
