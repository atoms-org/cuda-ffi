from cudaffi.memory import CudaDeviceMemory, CudaHostMemory, CudaManagedMemory


class TestMemory:
    def test_host(self) -> None:
        CudaHostMemory(1)

    def test_device(self) -> None:
        CudaDeviceMemory(1)

    def test_managed(self) -> None:
        CudaManagedMemory(1)
