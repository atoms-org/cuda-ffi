import ctypes
from typing import Any, cast

from ..memory import CudaMemory
from .base import AnyCType, CudaDataType


class CudaMemoryDataType(CudaDataType):
    def convert(self, data: Any, name: str) -> tuple[int, AnyCType] | None:
        if isinstance(data, CudaMemory):
            return (cast(int, data.nv_memory), ctypes.c_void_p)

        return None
