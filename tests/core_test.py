from cudaffi.core import CudaContext, CudaDevice, CudaStream


class TestDevice:
    def test_basic(self) -> None:
        CudaDevice()


class TestContext:
    def test_basic(self) -> None:
        CudaContext(CudaDevice())


class TestStream:
    def test_basic(self) -> None:
        CudaStream()
