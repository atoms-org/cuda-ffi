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
        if size_hint is None or size_hint < 1:
            size_hint = 4096

        ba = bytearray(size_hint)

        yield (ba, size_hint)

        # find the trailing "\0" to identify the real end of string
        for n in range(len(ba)):
            b = ba[n]
            if b == 0:
                break
        ba = ba[0:n]

        s = ba.decode(encoding="utf-8")

        return s
