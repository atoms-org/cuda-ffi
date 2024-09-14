import array
from typing import Any

from cuda import cuda

from ..memory import CudaDataType, CudaDeviceMemory
from ..utils import checkCudaErrors


class CudaArrayDataType(CudaDataType):
    def convert(self, name: str, data: Any) -> CudaDeviceMemory | None:
        if isinstance(data, array.array):
            sz = data.itemsize * len(data)
            mem = CudaDeviceMemory(sz)
            checkCudaErrors(cuda.cuMemcpyHtoD(mem.nv_device_memory, data, sz))
            return mem

        return None
