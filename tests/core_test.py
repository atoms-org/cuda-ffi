from cudaffi.core import CudaContext, CudaDevice, CudaStream, init


class TestDevice:
    def test_basic(self) -> None:
        CudaDevice()


class TestContext:
    def test_basic(self) -> None:
        CudaContext(CudaDevice())


class TestStream:
    def test_basic(self) -> None:
        CudaStream()


class TestInit:
    def test_basic(self) -> None:
        init(0)
