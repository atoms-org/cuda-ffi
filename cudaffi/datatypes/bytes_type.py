import ctypes
from typing import Any

from ..memory import AnyCType, CudaDataType, PointerOrHostMem, PointerOrPointerGenerator


class CudaBytesDataType(CudaDataType[bytes]):
    def is_type(self, data: Any) -> bool:
        return isinstance(data, bytes)

    def get_byte_size(self, data: bytes) -> int:
        return len(data)

    def get_ctype(self, data: bytes) -> AnyCType:
        return ctypes.c_void_p

    def encode(self, data: bytes) -> PointerOrHostMem | int:
        return (data, len(data))

    def decode(
        self, data: bytes | None = None, size_hint: int | None = None
    ) -> PointerOrPointerGenerator[bytes]:
        if data is not None:
            raise Exception("the type 'bytes' is immutable, cannot read into bytes")

        if size_hint is None:
            raise Exception("bytes needs size_hint for output")

        ba = bytearray(size_hint)

        yield (ba, size_hint)

        return bytes(ba)
