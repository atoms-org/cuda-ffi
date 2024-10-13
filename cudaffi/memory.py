from __future__ import annotations

from abc import ABC, abstractmethod

from cuda import cudart

from .device import init
from .utils import checkCudaErrorsAndReturn


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
