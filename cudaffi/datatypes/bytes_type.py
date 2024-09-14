from typing import Any

from cuda import cuda

from ..memory import CudaDataType, CudaDeviceMemory
from ..utils import checkCudaErrors


class CudaBytesDataType(CudaDataType):
    def convert(self, name: str, data: Any) -> CudaDeviceMemory | None:
        if isinstance(data, bytes):
            mem = CudaDeviceMemory(len(data))
            checkCudaErrors(cuda.cuMemcpyHtoD(mem.nv_device_memory, data, len(data)))
            return mem

        return None
