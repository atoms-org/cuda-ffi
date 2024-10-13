from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Buffer
from typing import Generator, TypeVar

from cuda import cudart

from .device import init
from .utils import checkCudaErrorsAndReturn

T = TypeVar("T")


class HostBuffer:
    def __init__(self, ptr: PointerOrHostMem) -> None:
        self.ptr: int | Buffer | cudart.cudaHostPtr
        self.size: int

        if isinstance(ptr, tuple):
            self.ptr = ptr[0]
            self.sz = ptr[1]
        elif isinstance(self.ptr, CudaHostMemory):
            self.mem = ptr
            self.ptr = ptr.nv_host_memory
            self.sz = ptr.size

    def to_host_nv_data(self) -> cudart.cudaHostPtr | Buffer | int:
        return self.ptr


class CudaMemory(ABC):
    def __init__(self, size: int):
        init()

        self.size = size

    # @abstractmethod
    # def free(self) -> None: ...

    @property
    @abstractmethod
    def dev_addr(self) -> cudart.cudaDevPtr: ...


class CudaHostMemory(CudaMemory):
    def __init__(self, size: int) -> None:
        super().__init__(size)

        flags = cudart.cudaHostAllocDefault
        self.nv_host_memory = checkCudaErrorsAndReturn(cudart.cudaHostAlloc(size, flags))

    @property
    def dev_addr(self) -> cudart.cudaDevPtr:
        raise Exception("attempting to use host memory as device address")


class CudaDeviceMemory(CudaMemory):
    def __init__(self, size: int) -> None:
        super().__init__(size)

        self.nv_device_memory = checkCudaErrorsAndReturn(cudart.cudaMalloc(size))

    @property
    def dev_addr(self) -> cudart.cudaDevPtr:
        return self.nv_device_memory


class CudaManagedMemory(CudaMemory):
    def __init__(self, size: int) -> None:
        super().__init__(size)

        flags = cudart.cudaMemAttachGlobal
        self.nv_managed_memory = checkCudaErrorsAndReturn(cudart.cudaMallocManaged(size, flags))

    @property
    def dev_addr(self) -> cudart.cudaDevPtr:
        return self.nv_managed_memory


# TODO: should we support collections.abc.memoryview everywhere we support Buffer?
# MemPointer = Buffer | int
MemPointer = int | Buffer
PointerAndSize = tuple[MemPointer, int]
PointerOrHostMem = PointerAndSize | CudaHostMemory
PointerGenerator = Generator[PointerOrHostMem, CudaDeviceMemory, T]
PointerOrPointerGenerator = PointerOrHostMem | PointerGenerator[T]
