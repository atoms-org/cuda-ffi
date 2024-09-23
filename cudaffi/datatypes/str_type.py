import ctypes
from collections.abc import Buffer
from typing import Any

from ..memory import AnyCType, CudaDataType, PointerOrPointerGenerator


class CudaStrDataType(CudaDataType[str]):
    default_direction = "input"

    def is_type(self, data: Any) -> bool:
        return isinstance(data, str)

    def get_byte_size(self, data: str) -> int:
        return len(data) + 1

    def get_ctype(self, data: str) -> AnyCType:
        return ctypes.c_void_p

    def encode(self, data: str) -> tuple[Buffer, int]:
        s = bytearray(data.encode())
        s.append(0)

        return (s, len(data) + 1)

    def decode(
        self, data: str | None = None, size_hint: int | None = None
    ) -> PointerOrPointerGenerator[str]:
        print("str decode")
        if size_hint is None or size_hint < 1:
            size_hint = 4096

        ba = bytearray(size_hint)

        yield (ba, size_hint)

        s = str(ba)

        return s
