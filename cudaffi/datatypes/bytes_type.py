import ctypes
from collections.abc import Buffer
from typing import Any, Generator

from ..args import CudaDataType


class CudaBytesDataType(CudaDataType[bytes]):
    def is_type(self, data: Any) -> bool:
        return isinstance(data, bytes)

    def get_byte_size(self, data: bytes) -> int:
        return len(data)

    def get_ctype(self, data: bytes) -> type[ctypes.c_void_p]:
        return ctypes.c_void_p

    def encode(self, data: bytes) -> tuple[Buffer, int]:
        return (data, len(data))

    def decode(
        self, data: bytes | None = None, size_hint: int | None = None
    ) -> Generator[tuple[Buffer, int], Any, bytes]:
        if data is not None:
            raise Exception("the type 'bytes' is immutable, cannot read into bytes")

        if size_hint is None:
            raise Exception("bytes needs size_hint for output")

        ba = bytearray(size_hint)

        yield (ba, size_hint)

        return bytes(ba)
