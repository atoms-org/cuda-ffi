from typing import Any

import numpy as np
from cuda import cuda

from ..memory import CudaDataType, CudaDeviceMemory
from ..utils import checkCudaErrors

AnyNpArray = np.ndarray[Any, Any]


class CudaNumpyDataType(CudaDataType[AnyNpArray]):
    def is_type(self, data: Any) -> bool:
        return isinstance(data, np.ndarray)

    def get_size(self, data: AnyNpArray) -> int:
        return data.nbytes

    def to_device(self, src: AnyNpArray, dst: CudaDeviceMemory) -> None:
        checkCudaErrors(cuda.cuMemcpyHtoD(dst.nv_device_memory, src.ctypes.data, src.nbytes))

    def to_host(self, src: CudaDeviceMemory, dst: AnyNpArray) -> None:
        checkCudaErrors(cuda.cuMemcpyDtoH(dst.ctypes.data, src.nv_device_memory, dst.nbytes))

        return None
