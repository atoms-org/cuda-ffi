from cudaffi.plan import CudaPlan, cuda_plan, parse_plan


class TestPlan:
    def test_decorator_with_args(self) -> None:
        @cuda_plan(msg="blah")
        def myfunc(foo: int) -> int:
            return 1

        myfunc(3)

    def test_decorator_without_args(self) -> None:
        @cuda_plan
        def myfunc(foo: int) -> int:
            return 1

        myfunc(3)

    def test_parse(self) -> None:
        def myfn(arg1: int, arg2: str) -> int:
            ret, out = mod1.kernel1(arg1, arg2)  # type: ignore
            ret2 = mod2.kernel2(ret, "this is a test", 1)  # type: ignore
            return ret2  # type: ignore

        parse_plan(myfn)


class TestCudaPlan:
    def test_call(self) -> None:
        def myfn(a: str, b: int) -> None:
            bob()  # type: ignore

        p = CudaPlan(myfn)

        assert len(p.steps) == 1
        assert p.steps[0].call_name == "bob"
        assert p.steps[0].call_module is None
        assert len(p.steps[0].output_vars) == 0
        assert len(p.steps[0].input_vars) == 0

    def test_attribute(self) -> None:
        def myfn(a: str, b: int) -> None:
            thingy.bob()  # type: ignore

        p = CudaPlan(myfn)

        assert len(p.steps) == 1
        assert p.steps[0].call_name == "bob"
        assert p.steps[0].call_module == "thingy"
        assert len(p.steps[0].output_vars) == 0
        assert len(p.steps[0].input_vars) == 0

    def test_assignment(self) -> None:
        def myfn(a: str, b: int) -> None:
            x = bob()  # type: ignore

        p = CudaPlan(myfn)

        assert len(p.steps) == 1
        assert p.steps[0].call_name == "bob"
        assert p.steps[0].call_module is None
        assert p.steps[0].input_arg_names == []
        assert p.steps[0].output_arg_names == ["x"]
        assert len(p.steps[0].input_vars) == 0

    def test_dereferencing(self) -> None:
        def myfn(a: str, b: int) -> None:
            x, y = bob()  # type: ignore

        p = CudaPlan(myfn)

        assert len(p.steps) == 1
        assert p.steps[0].call_name == "bob"
        assert p.steps[0].call_module is None
        assert p.steps[0].input_arg_names == []
        assert p.steps[0].output_arg_names == ["x", "y"]
        assert len(p.steps[0].input_vars) == 0

    def test_call_args(self) -> None:
        def myfn(a: str, b: int) -> None:
            bob(a, b)  # type: ignore

        p = CudaPlan(myfn)

        assert len(p.steps) == 1
        assert p.steps[0].call_name == "bob"
        assert p.steps[0].call_module is None
        assert p.steps[0].input_arg_names == ["a", "b"]
        assert p.steps[0].output_arg_names == []
        assert len(p.steps[0].output_vars) == 0
        assert len(p.steps[0].input_vars) == 0
        assert not p.steps[0].is_return

    def test_return_var(self) -> None:
        def myfn(a: str, b: int) -> str:
            return a

        p = CudaPlan(myfn)

        assert len(p.steps) == 1
        assert p.steps[0].call_name is None
        assert p.steps[0].call_module is None
        assert p.steps[0].is_return
        assert p.steps[0].output_arg_names == ["a"]

    def test_return_call(self) -> None:
        def myfn(a: str, b: int) -> str:
            return bob()  # type: ignore

        p = CudaPlan(myfn)

        assert len(p.steps) == 1
        assert p.steps[0].call_name == "bob"
        assert p.steps[0].call_module is None
        assert p.steps[0].is_return
        assert p.steps[0].input_arg_names == []
        assert p.steps[0].output_arg_names == []

    def test_return_call_with_module(self) -> None:
        def myfn(a: str, b: int) -> str:
            return beer.bob()  # type: ignore

        p = CudaPlan(myfn)

        assert len(p.steps) == 1
        assert p.steps[0].call_name == "bob"
        assert p.steps[0].call_module == "beer"
        assert p.steps[0].is_return
        assert p.steps[0].input_arg_names == []
        assert p.steps[0].output_arg_names == []

    def test_return_multiple(self) -> None:
        def myfn(a: str, b: int) -> tuple[str, int]:
            return a, b

        p = CudaPlan(myfn)

        assert len(p.steps) == 1
        assert p.steps[0].call_name is None
        assert p.steps[0].call_module is None
        assert p.steps[0].input_arg_names == ["a", "b"]
        assert p.steps[0].output_arg_names == ["a", "b"]
        assert p.steps[0].is_return

    # constant str
    # constant int
    # constant float
    # kwargs
