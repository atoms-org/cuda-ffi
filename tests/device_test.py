from cudaffi.core import CudaContext, CudaDevice, CudaStream, init


class TestDevice:
    def test_basic(self) -> None:
        CudaDevice()

    def test_attributes(self) -> None:
        dev = CudaDevice()
        print("name:", dev.name)
        print("compute capability:", dev.compute_capability)
        print("driver version:", dev.driver_version)


class TestContext:
    def test_basic(self) -> None:
        CudaContext(CudaDevice())


class TestStream:
    def test_basic(self) -> None:
        CudaStream()


class TestInit:
    def test_basic(self) -> None:
        init(0)
