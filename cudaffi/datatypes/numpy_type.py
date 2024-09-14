from typing import Any

import numpy as np
from cuda import cuda

from ..memory import CudaDataType, CudaDeviceMemory
from ..utils import checkCudaErrors


class CudaNumpyDataType(CudaDataType):
    def convert(self, name: str, data: Any) -> CudaDeviceMemory | None:
        if isinstance(data, np.ndarray):
            # TODO: only make contiguous if it's not already
            # you can check by seeing if arr.stride == arr.datasize
            arr = np.ascontiguousarray(data)
            mem = CudaDeviceMemory(arr.nbytes)
            checkCudaErrors(cuda.cuMemcpyHtoD(mem.nv_device_memory, arr, arr.nbytes))
            return mem

        return None
