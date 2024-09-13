from typing import Any

from cuda import cuda

from ..memory import CudaDataType, CudaDeviceMemory
from ..utils import checkCudaErrors


class CudaStrDataType(CudaDataType):
    def convert(self, name: str, data: Any) -> CudaDeviceMemory | None:
        if isinstance(data, str):
            s = bytearray(data.encode())
            s.append(0)
            mem = CudaDeviceMemory(len(s))
            checkCudaErrors(cuda.cuMemcpyHtoD(mem.nv_device_memory, s, len(s)))
            return mem

        return None
