import pytest

from cudaffi.module import CudaFunction, CudaModule
from cudaffi.plan import CudaPlan, CudaPlanException, CudaPlanVarType, cuda_plan


class TestPlan:
    def test_decorator_without_args(self) -> None:
        @cuda_plan
        def myfunc(foo: int) -> int:
            return foo

        myfunc(3)

        # make sure the function is now a CudaPlan
        assert isinstance(myfunc, CudaPlan)
        assert hasattr(myfunc, "fn_def_ast")  # type: ignore

        # make sure the wrapper worked
        assert myfunc.__name__ == "myfunc"
        assert hasattr(myfunc, "__wrapped__")

    def test_decorator_with_args(self) -> None:
        @cuda_plan()
        def myfunc(foo: int) -> int:
            return foo

        myfunc(3)

        # make sure the function is now a CudaPlan
        assert isinstance(myfunc, CudaPlan)
        assert hasattr(myfunc, "fn_def_ast")  # type: ignore

        # make sure the wrapper worked
        assert myfunc.__name__ == "myfunc"
        assert hasattr(myfunc, "__wrapped__")


class TestCudaPlanParsing:
    @pytest.fixture(autouse=True)
    def simple_mod(self) -> CudaModule:
        return CudaModule.from_file("tests/helpers/simple.cu")

    def test_function_definition(self) -> None:
        def myfn(a, b):  # type: ignore
            return a

        p = CudaPlan(myfn)

        assert p.inputs == [("a", "Any"), ("b", "Any")]
        assert p.output_type == "Any"

    def test_function_definition_with_types(self) -> None:
        def myfn(a: str, b: int) -> str:
            return a

        p = CudaPlan(myfn)

        assert p.inputs == [("a", "str"), ("b", "int")]
        assert p.output_type == "str"

    def test_function_definition_return_none(self) -> None:
        def myfn(a, b) -> None:  # type: ignore
            simple()  # type: ignore

        p = CudaPlan(myfn)

        assert p.output_type == "None"

    def test_call(self) -> None:
        def myfn(a: str, b: int) -> None:
            simple()  # type: ignore

        p = CudaPlan(myfn)

        assert len(p.steps) == 1
        assert isinstance(p.steps[0].call_fn, CudaFunction)
        assert p.steps[0].call_fn.name == "simple"
        assert len(p.steps[0].output_vars) == 0
        assert len(p.steps[0].input_vars) == 0

    # TODO: putting a comma between two statements turns it into a tuple
    # def test_comma(self) -> None:
    #     def myfn(a: str, b: int) -> None:
    #         foo(), bar()  # type: ignore

    #     p = CudaPlan(myfn)

    #     assert len(p.steps) == 1
    #     assert p.steps[0].call_name == "bob"
    #     assert p.steps[0].call_module is None
    #     assert len(p.steps[0].output_vars) == 0
    #     assert len(p.steps[0].input_vars) == 0
    #     print("output type", p.output_type)

    def test_attribute(self, simple_mod: CudaModule) -> None:
        def myfn(a: str, b: int) -> None:
            thingy.simple()  # type: ignore

        p = CudaPlan(myfn, modules={"thingy": simple_mod})

        assert len(p.vars) == 2
        assert len(p.steps) == 1
        assert isinstance(p.steps[0].call_fn, CudaFunction)
        assert p.steps[0].call_fn.name == "simple"
        assert len(p.steps[0].output_vars) == 0
        assert len(p.steps[0].input_vars) == 0

    def test_assignment(self) -> None:
        def myfn(a: str, b: int) -> None:
            x = simple()  # type: ignore

        p = CudaPlan(myfn)

        assert len(p.vars) == 3
        assert len(p.steps) == 1
        assert isinstance(p.steps[0].call_fn, CudaFunction)
        assert p.steps[0].call_fn.name == "simple"
        assert len(p.steps[0].input_vars) == 0
        assert len(p.steps[0].output_vars) == 1
        assert p.steps[0].output_vars[0].name == "x"
        assert p.steps[0].output_vars[0].type == CudaPlanVarType.arg
        assert len(p.steps[0].input_vars) == 0

    def test_dereferencing(self) -> None:
        def myfn(a: str, b: int) -> None:
            x, y = simple()  # type: ignore

        p = CudaPlan(myfn)

        assert len(p.vars) == 4
        assert len(p.steps) == 1
        assert isinstance(p.steps[0].call_fn, CudaFunction)
        assert p.steps[0].call_fn.name == "simple"
        assert len(p.steps[0].input_vars) == 0
        assert len(p.steps[0].output_vars) == 2
        assert p.steps[0].output_vars[0].name == "x"
        assert p.steps[0].output_vars[0].type == CudaPlanVarType.arg
        assert p.steps[0].output_vars[1].name == "y"
        assert p.steps[0].output_vars[1].type == CudaPlanVarType.arg
        assert len(p.steps[0].input_vars) == 0

    def test_call_args(self) -> None:
        def myfn(a: str, b: int) -> None:
            simple(a, b)  # type: ignore

        p = CudaPlan(myfn)

        assert len(p.vars) == 2
        assert len(p.steps) == 1
        assert isinstance(p.steps[0].call_fn, CudaFunction)
        assert p.steps[0].call_fn.name == "simple"
        assert len(p.steps[0].input_vars) == 2
        assert p.steps[0].input_vars[0].name == "a"
        assert p.steps[0].input_vars[0].type == CudaPlanVarType.arg
        assert p.steps[0].input_vars[1].name == "b"
        assert p.steps[0].input_vars[1].type == CudaPlanVarType.arg
        assert len(p.steps[0].output_vars) == 0
        assert not p.steps[0].is_return

    def test_call_args_constant(self) -> None:
        def myfn(a: str, b: int) -> None:
            simple(42, 3.14, "foo", True)  # type: ignore

        p = CudaPlan(myfn)

        assert len(p.vars) == 6
        assert len(p.steps) == 1
        assert isinstance(p.steps[0].call_fn, CudaFunction)
        assert p.steps[0].call_fn.name == "simple"
        assert len(p.steps[0].input_vars) == 4
        assert p.steps[0].input_vars[0].type == CudaPlanVarType.constant
        assert p.steps[0].input_vars[0].name == "42"
        assert p.steps[0].input_vars[0].val == 42
        assert p.steps[0].input_vars[1].type == CudaPlanVarType.constant
        assert p.steps[0].input_vars[1].name == "3.14"
        assert p.steps[0].input_vars[1].val == 3.14
        assert p.steps[0].input_vars[2].type == CudaPlanVarType.constant
        assert p.steps[0].input_vars[2].name == "foo"
        assert p.steps[0].input_vars[2].val == "foo"
        assert p.steps[0].input_vars[3].type == CudaPlanVarType.constant
        assert p.steps[0].input_vars[3].name == "True"
        assert p.steps[0].input_vars[3].val == True
        assert len(p.steps[0].output_vars) == 0
        assert not p.steps[0].is_return

    def test_call_kwargs(self) -> None:
        def myfn(a: str, b: int) -> None:
            simple(a, block=(1, 1, 1), grid=(1, 1, 1))  # type: ignore

        p = CudaPlan(myfn)

        assert len(p.steps) == 1
        assert isinstance(p.steps[0].call_fn, CudaFunction)
        assert p.steps[0].call_fn.name == "simple"
        assert len(p.steps[0].input_vars) == 1
        assert p.steps[0].input_vars[0].name == "a"
        assert p.steps[0].input_vars[0].type == CudaPlanVarType.arg
        assert len(p.steps[0].input_kwvars.keys()) == 2
        assert "block" in p.steps[0].input_kwvars
        assert p.steps[0].input_kwvars["block"].name == "(1, 1, 1)"
        assert p.steps[0].input_kwvars["block"].type == CudaPlanVarType.constant
        assert p.steps[0].input_kwvars["block"].val == (1, 1, 1)
        assert "grid" in p.steps[0].input_kwvars
        assert p.steps[0].input_kwvars["grid"].name == "(1, 1, 1)"
        assert p.steps[0].input_kwvars["grid"].type == CudaPlanVarType.constant
        assert p.steps[0].input_kwvars["grid"].val == (1, 1, 1)
        assert len(p.steps[0].output_vars) == 0
        assert not p.steps[0].is_return

    def test_return_var(self) -> None:
        def myfn(a: str, b: int) -> str:
            return a

        p = CudaPlan(myfn)

        assert len(p.steps) == 1
        assert p.steps[0].call_fn is None
        assert p.steps[0].is_return
        assert len(p.steps[0].input_vars) == 1
        assert p.steps[0].input_vars[0].name == "a"
        assert p.steps[0].input_vars[0].type == CudaPlanVarType.arg

    def test_return_call(self) -> None:
        def myfn(a: str, b: int) -> str:
            return simple()  # type: ignore

        p = CudaPlan(myfn)

        assert len(p.steps) == 1
        assert isinstance(p.steps[0].call_fn, CudaFunction)
        assert p.steps[0].call_fn.name == "simple"
        assert p.steps[0].is_return
        assert len(p.steps[0].input_vars) == 0
        assert len(p.steps[0].output_vars) == 0

    def test_return_call_with_module(self, simple_mod: CudaModule) -> None:
        def myfn(a: str, b: int) -> str:
            return beer.simple()  # type: ignore

        p = CudaPlan(myfn, modules={"beer": simple_mod})

        assert len(p.steps) == 1
        assert isinstance(p.steps[0].call_fn, CudaFunction)
        assert p.steps[0].call_fn.name == "simple"
        assert p.steps[0].is_return
        assert len(p.steps[0].input_vars) == 0
        assert len(p.steps[0].output_vars) == 0

    def test_return_multiple(self) -> None:
        def myfn(a: str, b: int) -> tuple[str, int]:
            return a, b

        p = CudaPlan(myfn)

        assert len(p.vars) == 2
        assert len(p.steps) == 1
        assert p.steps[0].call_fn is None
        assert len(p.steps[0].input_vars) == 2
        assert p.steps[0].input_vars[0].name == "a"
        assert p.steps[0].input_vars[0].type == CudaPlanVarType.arg
        assert p.steps[0].input_vars[1].name == "b"
        assert p.steps[0].input_vars[1].type == CudaPlanVarType.arg
        assert len(p.steps[0].output_vars) == 2
        assert p.steps[0].is_return
        assert p.output_type == "tuple[str, int]"

    def test_return_constant(self) -> None:
        def myfn(a: str, b: int) -> int:
            return 42

        p = CudaPlan(myfn)

        assert len(p.vars) == 3
        assert len(p.steps) == 1
        assert p.steps[0].call_fn is None
        assert len(p.steps[0].input_vars) == 0
        assert len(p.steps[0].output_vars) == 1
        assert p.steps[0].output_vars[0].name == "42"
        assert p.steps[0].output_vars[0].type == CudaPlanVarType.constant
        assert p.steps[0].output_vars[0].val == 42
        assert p.steps[0].is_return
        assert p.output_type == "int"

    def test_steps_after_return(self) -> None:
        def myfn(a: str, b: int) -> int:
            return 42
            simple()  # type: ignore

        with pytest.raises(CudaPlanException) as e:
            CudaPlan(myfn)

        assert e.value.message == "statements found after return in CudaPlan"


class TestCudaPlan:
    @pytest.fixture(autouse=True)
    def simple_mod(self) -> CudaModule:
        return CudaModule.from_file("tests/helpers/simple.cu")

    def test_to_graph_basic(self, simple_mod: CudaModule) -> None:
        mymod = CudaModule.from_file("tests/helpers/simple.cu")

        @cuda_plan(modules={"mymod": simple_mod})
        def myfn() -> None:
            mymod.simple()
