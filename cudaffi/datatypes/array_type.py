from array import array
from typing import Any

from ..memory import CudaDataType, PointerOrHostMem, PointerOrPointerGenerator

AnyArray = array[Any]


class CudaArrayDataType(CudaDataType[AnyArray]):
    def is_type(self, data: Any) -> bool:
        return isinstance(data, bytearray)

    def get_byte_size(self, data: AnyArray) -> int:
        return len(data)

    def encode(self, data: AnyArray) -> PointerOrHostMem | int:
        return (data, len(data))

    def decode(
        self, data: AnyArray | None = None, size_hint: int | None = None
    ) -> PointerOrPointerGenerator[AnyArray]:
        if data is None:
            if size_hint is None:
                raise Exception("need either bytearray or size hint")
            data = array("Q", [0] * size_hint)  # TODO: real types

        return (data, len(data))
