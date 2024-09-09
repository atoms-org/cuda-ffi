import ctypes
from typing import Any

from .base import AnyCType, CudaDataType


class CudaIntDataType(CudaDataType):
    def convert(self, data: Any, name: str) -> tuple[int, AnyCType] | None:
        if isinstance(data, int):
            return (data, ctypes.c_uint)

        return None
