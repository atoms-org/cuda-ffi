from cudaffi.module import CudaModule


class TestModule:
    def test_exists(self) -> None:
        CudaModule()
