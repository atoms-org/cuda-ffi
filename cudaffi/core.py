from __future__ import annotations

from typing import NewType, cast

from cuda import cuda

from .utils import checkCudaErrors

NvDevice = NewType("NvDevice", object)  # cuda.CUdevice
NvContext = NewType("NvContext", object)  # cuda.CUcontext
NvStream = NewType("NvStream", object)  # cuda.CUstream

_default_device: CudaDevice | None = None
_default_context: CudaContext | None = None
_default_stream: CudaStream | None = None
_initialized = False


class CudaInitializationException(Exception):
    pass


def init(flags: int = 0) -> None:
    global _initialized
    if not _initialized:
        checkCudaErrors(cuda.cuInit(flags))
        _initialized = True
        dev = CudaDevice(0)
        CudaDevice.set_default(dev)
        ctx = CudaContext(dev)
        CudaContext.set_default(ctx)
        stream = CudaStream()
        CudaStream.set_default(stream)

        from .datatypes import init as init_datatype

        init_datatype()


class CudaStream:
    def __init__(self, flags: int = cuda.CUstream_flags.CU_STREAM_DEFAULT) -> None:
        init()

        self.nv_stream: NvStream = checkCudaErrors(cuda.cuStreamCreate(flags))

    # def __del__(self) -> None:
    #     self.synchronize()
    #     checkCudaErrors(cuda.cuStreamDestroy(self.nv_stream))

    def synchronize(self) -> None:
        checkCudaErrors(cuda.cuStreamSynchronize(self.nv_stream))

    @staticmethod
    def set_default(stream: CudaStream) -> None:
        global _default_stream
        _default_stream = stream

    @staticmethod
    def get_default() -> CudaStream:
        global _default_stream
        if _default_stream is None:
            init()
        assert _default_stream is not None
        return _default_stream


class CudaContext:
    def __init__(self, dev: "CudaDevice") -> None:
        init()

        self.nv_context: NvContext = checkCudaErrors(cuda.cuCtxCreate(0, dev.nv_device))

    def __del__(self) -> None:
        checkCudaErrors(cuda.cuCtxDestroy(self.nv_context))

    @staticmethod
    def set_default(ctx: CudaContext) -> None:
        global _default_context
        _default_context = ctx

    @staticmethod
    def get_default() -> CudaContext:
        global _default_context
        if _default_context is None:
            init()
        assert _default_context is not None
        return _default_context


class CudaDevice:
    def __init__(self, device_id: int = 0) -> None:
        init()

        self.dev_id = device_id
        self.contexts: list[CudaContext] = []
        self.streams: list[CudaStream] = []

        # Retrieve handle for device 0
        self.nv_device: NvDevice = checkCudaErrors(cuda.cuDeviceGet(device_id))

    def create_context(self) -> CudaContext:
        # Create context
        return CudaContext(self)

    def create_stream(self) -> CudaStream:
        return CudaStream()

    @property
    def name(self) -> str:
        name: bytes = checkCudaErrors(cuda.cuDeviceGetName(512, self.nv_device))
        return name.decode()

    @property
    def default_context(self) -> CudaContext:
        # TODO: cuDevicePrimaryCtxRetain?
        print("getting default context")
        if len(self.contexts) == 0:
            self.contexts.append(self.create_context())

        return self.contexts[0]

    @property
    def default_stream(self) -> CudaStream:
        if len(self.streams) == 0:
            self.streams.append(self.create_stream())

        return self.streams[0]

    @property
    def compute_capability(self) -> tuple[int, int]:
        # Derive target architecture for device 0
        major = checkCudaErrors(
            cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self.nv_device
            )
        )
        minor = checkCudaErrors(
            cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self.nv_device
            )
        )
        return (major, minor)

    @property
    def driver_version(self) -> tuple[int, int]:
        version_num = checkCudaErrors(cuda.cuDriverGetVersion())
        major = version_num // 1000
        minor = (version_num - (major * 1000)) // 10
        return (major, minor)

    @staticmethod
    def set_default(dev: CudaDevice) -> None:
        global _default_device
        _default_device = dev

    @staticmethod
    def get_default() -> CudaDevice:
        global _default_device
        if _default_device is None:
            init()
        assert _default_device is not None
        return _default_device

    @staticmethod
    def count() -> int:
        init()

        return cast(int, checkCudaErrors(cuda.cuDeviceGetCount()))
