import ctypes
from array import array
from collections.abc import Buffer
from typing import Any

from ..args import CudaDataType

AnyArray = array[Any]


class CudaArrayDataType(CudaDataType[AnyArray]):
    def is_type(self, data: Any) -> bool:
        return isinstance(data, bytearray)

    def get_byte_size(self, data: AnyArray) -> int:
        return len(data)

    def get_ctype(self, data: AnyArray) -> type[ctypes.c_void_p]:
        return ctypes.c_void_p

    def encode(self, data: AnyArray) -> tuple[Buffer, int]:
        return (data, len(data))

    def decode(
        self, data: AnyArray | None = None, size_hint: int | None = None
    ) -> tuple[Buffer, int]:
        if data is None:
            if size_hint is None:
                raise Exception("need either bytearray or size hint")
            data = array("Q", [0] * size_hint)  # TODO: real types

        return (data, len(data))
