import pytest

from cudaffi.module import (
    CudaCompilationError,
    CudaCompilationWarning,
    CudaFunction,
    CudaFunctionNameNotFound,
    CudaModule,
)


class TestModule:
    def test_exists(self) -> None:
        CudaModule.from_file("tests/helpers/simple.cu")


class TestFunction:
    def test_basic(self) -> None:
        mod = CudaModule(
            """
        __global__ void thingy() {
          printf("this is a test\\n");
        }
        """
        )
        fn = mod.get_function("thingy")
        assert isinstance(fn, CudaFunction)
        mod.thingy()

    def test_from_file(self) -> None:
        mod = CudaModule.from_file("tests/helpers/simple.cu")
        fn = mod.get_function("simple")
        assert isinstance(fn, CudaFunction)
        mod.simple()

    def test_one_arg(self) -> None:
        mod = CudaModule.from_file("tests/helpers/one_arg.cu")
        fn = mod.get_function("one")
        assert isinstance(fn, CudaFunction)
        mod.one(1)

    def test_compilation_error(self) -> None:
        with pytest.raises(CudaCompilationError, match='error: expected a ";"'):
            CudaModule(
                """
            __global__ void thingy() {
            printf("this is a test\\n") // missing semicolon
            }
            """
            )

    def test_compilation_warning(self) -> None:
        with pytest.warns(
            CudaCompilationWarning, match="the format string requires additional arguments"
        ):
            CudaModule(
                """
            __global__ void thingy() {
            printf("this is a test %d\\n"); // missing argument
            }
            """
            )

    def test_wrong_name(self) -> None:
        mod = CudaModule.from_file("tests/helpers/one_arg.cu")
        with pytest.raises(CudaFunctionNameNotFound):
            mod.doesnotexist(1)
