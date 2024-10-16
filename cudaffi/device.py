from __future__ import annotations

from cuda import cuda

from .utils import checkCudaErrorsAndReturn, checkCudaErrorsNoReturn

_default_device: CudaDevice | None = None
_default_context: CudaContext | None = None
_default_stream: CudaStream | None = None
_initialized = False


class CudaInitializationException(Exception):
    pass


def init(flags: int = 0, force: bool = False) -> None:
    global _initialized
    if not _initialized or force:
        checkCudaErrorsNoReturn(cuda.cuInit(flags))
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

        self.nv_stream = checkCudaErrorsAndReturn(cuda.cuStreamCreate(flags))

    # def __del__(self) -> None:
    #     self.synchronize()
    #     checkCudaErrors(cuda.cuStreamDestroy(self.nv_stream))

    def synchronize(self) -> None:
        checkCudaErrorsNoReturn(cuda.cuStreamSynchronize(self.nv_stream))

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

        self.nv_context = checkCudaErrorsAndReturn(cuda.cuCtxCreate(0, dev.nv_device))

    def __del__(self) -> None:
        checkCudaErrorsNoReturn(cuda.cuCtxDestroy(self.nv_context))

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


class UnknownAttribute(Exception):
    pass


class CudaDevice:
    def __init__(self, device_id: int = 0) -> None:
        init()

        self.dev_id = device_id
        self.contexts: list[CudaContext] = []
        self.streams: list[CudaStream] = []

        # Retrieve handle for device 0
        self.nv_device = checkCudaErrorsAndReturn(cuda.cuDeviceGet(device_id))

    def create_context(self) -> CudaContext:
        # Create context
        return CudaContext(self)

    def create_stream(self) -> CudaStream:
        return CudaStream()

    def get_attribute(self, attr: str) -> int:
        attr = attr.upper()
        if not attr.startswith("CU_DEVICE_ATTRIBUTE_"):
            attr = "CU_DEVICE_ATTRIBUTE_" + attr

        if not hasattr(cuda.CUdevice_attribute, attr):
            raise UnknownAttribute(f"unknown CUDA device attribute: '{attr}'")

        attr_val = getattr(cuda.CUdevice_attribute, attr)
        val = checkCudaErrorsAndReturn(cuda.cuDeviceGetAttribute(attr_val, self.nv_device))

        return val

    @property
    def all_attribute_names(self) -> list[str]:
        attr_names = [
            v.name.replace("CU_DEVICE_ATTRIBUTE_", "").lower() for v in cuda.CUdevice_attribute
        ]
        # XXX: remove the last one, which is "max"
        # XXX: CUDA Python seems to have an off-by-one error where requesting
        # CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED results in an error
        return attr_names[:-2]

    @property
    def name(self) -> str:
        name_buf: bytes = checkCudaErrorsAndReturn(cuda.cuDeviceGetName(512, self.nv_device))
        strlen = 0
        for strlen in range(len(name_buf)):
            if name_buf[strlen] == 0:
                break

        name_str = name_buf.decode()

        return name_str[:strlen]

    @property
    def default_context(self) -> CudaContext:
        # TODO: cuDevicePrimaryCtxRetain?
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
        major = self.get_attribute("compute_capability_major")
        minor = self.get_attribute("compute_capability_minor")
        return (major, minor)

    @property
    def driver_version(self) -> tuple[int, int]:
        version_num = checkCudaErrorsAndReturn(cuda.cuDriverGetVersion())
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

        return checkCudaErrorsAndReturn(cuda.cuDeviceGetCount())
