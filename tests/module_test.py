from cudaffi.module import CudaModule


class TestModule:
    def test_exists(self) -> None:
        CudaModule.from_file("tests/helpers/simple.cu")


class TestFunction:
    def test_basic(self) -> None:
        mod = CudaModule.from_file("tests/helpers/simple.cu")
        fn = mod.get_function("simple")
