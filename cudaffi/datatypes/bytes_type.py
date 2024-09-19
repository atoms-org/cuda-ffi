from collections.abc import Buffer
from typing import Any

from cuda import cuda

from ..memory import CudaDataType, CudaDeviceMemory
from ..utils import checkCudaErrors


class CudaBytesDataType(CudaDataType):
    def is_type(self, data: Any) -> bool:
        return isinstance(data, bytes)

    def get_size(self, data: bytes) -> int:
        return len(data)

    def get_ptr(self, data: bytes) -> Buffer:
        return data

    def to_device(self, src: bytes, dst: CudaDeviceMemory) -> None:
        checkCudaErrors(cuda.cuMemcpyHtoD(dst.nv_device_memory, src, len(src)))

    def to_host(self, src: CudaDeviceMemory, dst: bytes) -> None:
        checkCudaErrors(cuda.cuMemcpyDtoH(dst, src.nv_device_memory, len(dst)))

    def convert(self, name: str, data: Any) -> CudaDeviceMemory | None:
        if isinstance(data, bytes):
            mem = CudaDeviceMemory(len(data))
            checkCudaErrors(cuda.cuMemcpyHtoD(mem.nv_device_memory, data, len(data)))
            return mem

        return None
