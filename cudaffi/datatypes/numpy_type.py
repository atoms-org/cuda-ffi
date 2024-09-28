import ctypes
from typing import Any

import numpy as np

from ..args import CudaDataType

AnyNpArray = np.ndarray[Any, Any]


class CudaNumpyDataType(CudaDataType[AnyNpArray]):
    def is_type(self, data: Any) -> bool:
        return isinstance(data, bytearray)

    def get_byte_size(self, data: AnyNpArray) -> int:
        return len(data)

    def get_ctype(self, data: AnyNpArray) -> type[ctypes.c_void_p]:
        return ctypes.c_void_p

    def encode(self, arr: AnyNpArray) -> tuple[int, int]:
        data = arr.ctypes.data
        return (data, len(arr))

    def decode(
        self, arr: AnyNpArray | None = None, size_hint: int | None = None
    ) -> tuple[int, int]:
        if arr is None:
            if size_hint is None:
                raise Exception("numpy needs size_hint if array isn't provided")
            arr = np.array(size_hint)

        return (arr.ctypes.data, len(arr))
