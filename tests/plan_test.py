from cudaffi.plan import cuda_plan, parse_plan


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
