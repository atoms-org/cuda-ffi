from __future__ import annotations

import ctypes
from abc import ABC, abstractmethod
from collections.abc import Buffer
from typing import TYPE_CHECKING, Any, Generator, Generic, NewType, TypeAlias, TypeVar

from cuda import cudart

from .core import init
from .utils import checkCudaErrors

NvDeviceMemory = NewType("NvDeviceMemory", object)  # cuda.CUdeviceptr
NvHostMemory = NewType("NvHostMemory", object)
NvManagedMemory = NewType("NvManagedMemory", object)


data_type_registry: DataTypeRegistry = {}
DataType = TypeVar("DataType")

# XXX - https://mypy.readthedocs.io/en/stable/runtime_troubles.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
if TYPE_CHECKING:
    AnyCType: TypeAlias = type[ctypes._SimpleCData[Any]]
else:
    AnyCType: TypeAlias = ctypes._SimpleCData


class CudaDataType(ABC, Generic[DataType]):
    aliases: list[str] = list()
    default_direction: str = "inout"

    def __init__(self, name: str) -> None:
        self.name = str

    @abstractmethod
    def is_type(self, data: Any) -> bool: ...

    @abstractmethod
    def get_byte_size(self, data: DataType) -> int: ...

    @abstractmethod
    def get_ctype(self, data: DataType) -> AnyCType: ...

    @abstractmethod
    def encode(self, data: DataType) -> PointerOrHostMem | int: ...

    @abstractmethod
    def decode(
        self, data: DataType | None = None, size_hint: int | None = None
    ) -> PointerOrPointerGenerator[DataType]: ...

    @staticmethod
    def register(
        name: str,
        DataType: type[CudaDataType[Any]],
        force: bool = False,
        is_alias: bool = False,
    ) -> None:
        global data_type_registry
        if name in data_type_registry and not force:
            raise Exception(f"'{name}' already exists as a registered CudaDataType")

        dt = DataType(name)
        data_type_registry[name] = dt
        if hasattr(dt, "aliases") and not is_alias:
            for alias in dt.aliases:
                DataType.register(alias, DataType, is_alias=True)

    @staticmethod
    def get_registry() -> DataTypeRegistry:
        global data_type_registry
        return data_type_registry


DataTypeRegistry = dict[str, CudaDataType[Any]]


class CudaMemory(ABC):
    def __init__(self, size: int):
        init()

        self.size = size
        self.ctype = ctypes.c_void_p

    # allocates the memory

    # @abstractmethod
    # def set(self, dest_data: Any) -> None: ...

    # copies from memory to the target destination using a datatype converter

    # @abstractmethod
    # def get(self, src_data: Any) -> None: ...

    # copies data from src to this memory location using a datatype converter

    # @abstractmethod
    # def free(self) -> None: ...

    # deallocates the memory

    @property
    @abstractmethod
    def dev_addr(self) -> NvDeviceMemory | NvManagedMemory: ...

    def copy_to(self, dst_mem: CudaMemory) -> None:
        pass

    # copies between memory locations

    def copy_from(self, src_mem: CudaMemory) -> None:
        pass

    # copies between memory locations

    @staticmethod
    def memcpy(src: CudaMemory, dst: CudaMemory) -> None:
        pass


class CudaHostMemory(CudaMemory):
    def __init__(self, size: int) -> None:
        super().__init__(size)

        flags = cudart.cudaHostAllocDefault
        self.nv_host_memory: NvHostMemory = checkCudaErrors(cudart.cudaHostAlloc(size, flags))
        print("self.nv_host_memory", self.nv_host_memory.__class__)

    @property
    def dev_addr(self) -> NvDeviceMemory:
        raise Exception("attempting to use host memory as device address")


class CudaDeviceMemory(CudaMemory):
    def __init__(self, size: int) -> None:
        super().__init__(size)

        self.nv_device_memory: NvDeviceMemory = checkCudaErrors(cudart.cudaMalloc(size))
        print("self.nv_device_memory", self.nv_device_memory.__class__)

    @property
    def dev_addr(self) -> NvDeviceMemory:
        return self.nv_device_memory


class CudaManagedMemory(CudaMemory):
    def __init__(self, size: int) -> None:
        super().__init__(size)

        flags = cudart.cudaMemAttachGlobal
        self.nv_managed_memory: NvManagedMemory = checkCudaErrors(
            cudart.cudaMallocManaged(size, flags)
        )
        print("self.nv_managed_memory", self.nv_managed_memory.__class__)

    @property
    def dev_addr(self) -> NvManagedMemory:
        return self.nv_managed_memory


# TODO: should we support collections.abc.memoryview everywhere we support Buffer?
MemPointer = Buffer | int
PointerAndSize = tuple[MemPointer, int]
PointerOrHostMem = PointerAndSize | CudaHostMemory
PointerGenerator = Generator[PointerOrHostMem, CudaDeviceMemory, DataType]
PointerOrPointerGenerator = PointerOrHostMem | PointerGenerator[DataType]
