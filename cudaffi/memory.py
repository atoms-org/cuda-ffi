from __future__ import annotations

import ctypes
from abc import ABC, abstractmethod
from typing import Any, NewType

from cuda import cudart

from .args import CudaArgType
from .core import init
from .utils import checkCudaErrors

NvDeviceMemory = NewType("NvDeviceMemory", object)  # cuda.CUdeviceptr
NvHostMemory = NewType("NvHostMemory", object)
NvManagedMemory = NewType("NvManagedMemory", object)


data_type_registry: DataTypeRegistry = {}
AnyCType = type[ctypes.c_uint] | type[ctypes.c_void_p]


class CudaDataType(ABC):
    def __init__(self, name: str) -> None:
        self.name = str

    @abstractmethod
    def convert(self, type: str, data: Any) -> CudaMemory | None: ...

    @staticmethod
    def register(name: str, DataType: type[CudaDataType], force: bool = False) -> None:
        global data_type_registry
        if name in data_type_registry and not force:
            raise Exception(f"'{name}' already exists as a registered CudaDataType")

        data_type_registry[name] = DataType(name)

    @staticmethod
    def get_registry() -> DataTypeRegistry:
        global data_type_registry
        return data_type_registry


DataTypeRegistry = dict[str, CudaDataType]


class CudaDataConversionError(Exception):
    def __init__(self, data: Any, arg_type: CudaArgType | None, msg: str):
        super().__init__(msg)
        self.data = data
        self.arg_type = arg_type


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

    @staticmethod
    def from_any(data: Any, arg_type: CudaArgType | None = None) -> CudaMemory:
        init()

        mem: CudaMemory | None = None
        data_type_registry = CudaDataType.get_registry()
        if arg_type is not None:
            arg_type_str = arg_type.type
            if arg_type_str not in data_type_registry:
                raise Exception(f"'{arg_type_str}' is not a registered data type")  # TODO
            datatype = data_type_registry[arg_type_str]

            mem = datatype.convert(arg_type_str, data)

        for type in data_type_registry:
            mem = data_type_registry[type].convert(type, data)
            if mem is not None:
                break

        if mem is None:
            if arg_type is not None:
                raise CudaDataConversionError(
                    data, arg_type, f"data could not be converted to '{arg_type.type}'"
                )
            else:
                raise CudaDataConversionError(data, arg_type, f"converter not found for data")

        return mem


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
