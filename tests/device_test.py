from cudaffi.device import CudaContext, CudaDevice, CudaStream, init


class TestDevice:
    def test_basic(self) -> None:
        CudaDevice()

    def test_attributes(self) -> None:
        dev = CudaDevice()

        # don't know if this is true, but if it's not the testing isn't going to
        # get very far on a device without memory pools :)
        assert dev.get_attribute("CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED") == 1
        assert dev.get_attribute("MEMORY_POOLS_SUPPORTED") == 1
        assert dev.get_attribute("cu_device_attribute_memory_pools_supported") == 1
        assert dev.get_attribute("memory_pools_supported") == 1

        # useful for testing CI
        print("name:", dev.name)  # noqa: T201
        print("compute capability:", dev.compute_capability)  # noqa: T201
        print("driver version:", dev.driver_version)  # noqa: T201
        for attr in dev.all_attribute_names:
            print(f"{attr}: {dev.get_attribute(attr)}")  # noqa: T201


class TestContext:
    def test_basic(self) -> None:
        CudaContext(CudaDevice())


class TestStream:
    def test_basic(self) -> None:
        CudaStream()


class TestInit:
    def test_basic(self) -> None:
        init(0)
