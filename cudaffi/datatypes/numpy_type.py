from typing import Any

import numpy as np

from ..memory import CudaDataType, PointerOrHostMem, PointerOrPointerGenerator

AnyNpArray = np.ndarray[Any, Any]


class CudaNumpyDataType(CudaDataType[AnyNpArray]):
    def is_type(self, data: Any) -> bool:
        return isinstance(data, bytearray)

    def get_byte_size(self, data: AnyNpArray) -> int:
        return len(data)

    def encode(self, arr: AnyNpArray) -> PointerOrHostMem:
        data = arr.ctypes.data
        return (data, len(arr))

    def decode(
        self, arr: AnyNpArray | None = None, size_hint: int | None = None
    ) -> PointerOrPointerGenerator[AnyNpArray]:
        if arr is None:
            if size_hint is None:
                raise Exception("numpy needs size_hint if array isn't provided")
            arr = np.array(size_hint)

        return (arr.ctypes.data, len(arr))
