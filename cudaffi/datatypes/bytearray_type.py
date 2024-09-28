import ctypes
from collections.abc import Buffer
from typing import Any

from ..args import CudaDataType


class CudaByteArrayDataType(CudaDataType[bytearray]):
    def is_type(self, data: Any) -> bool:
        return isinstance(data, bytearray)

    def get_byte_size(self, data: bytearray) -> int:
        return len(data)

    def get_ctype(self, data: bytearray) -> type[ctypes.c_void_p]:
        return ctypes.c_void_p

    def encode(self, data: bytearray) -> tuple[Buffer, int]:
        return (data, len(data))

    def decode(
        self, data: bytearray | None = None, size_hint: int | None = None
    ) -> tuple[Buffer, int]:
        if data is None:
            if size_hint is None:
                raise Exception("need either bytearray or size hint")
            data = bytearray(size_hint)

        return (data, len(data))
