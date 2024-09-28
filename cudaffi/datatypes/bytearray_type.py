import ctypes
from typing import Any

from ..memory import AnyCType, CudaDataType, PointerOrHostMem, PointerOrPointerGenerator


class CudaByteArrayDataType(CudaDataType[bytearray]):
    def is_type(self, data: Any) -> bool:
        return isinstance(data, bytearray)

    def get_byte_size(self, data: bytearray) -> int:
        return len(data)

    def get_ctype(self, data: bytearray) -> AnyCType:
        return ctypes.c_void_p

    def encode(self, data: bytearray) -> PointerOrHostMem | int:
        return (data, len(data))

    def decode(
        self, data: bytearray | None = None, size_hint: int | None = None
    ) -> PointerOrPointerGenerator[bytearray]:
        if data is None:
            if size_hint is None:
                raise Exception("need either bytearray or size hint")
            data = bytearray(size_hint)

        return (data, len(data))
