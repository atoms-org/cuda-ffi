import ctypes
import warnings
from typing import Any

from ..memory import AnyCType, CudaDataType, PointerOrPointerGenerator


class CudaIntDataType(CudaDataType[int]):
    default_direction = "input"
    aliases = ["uint"]

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.signed = not name.startswith("u")
        self.byte_size = 4

    def is_type(self, data: Any) -> bool:
        return isinstance(data, int)

    def get_byte_size(self, data: int) -> int:
        return self.byte_size

    def get_ctype(self, data: int) -> AnyCType:
        return ctypes.c_int

    def encode(self, data: int) -> int:
        num_bits = self.byte_size * 8
        if self.signed:
            num_bits -= 1
            max_size = (2**num_bits) - 1
            min_size = 2**num_bits
        else:
            max_size = (2**num_bits) - 1
            min_size = 0

        if data > max_size:
            warnings.warn("int too big for size")

        if data < min_size:
            warnings.warn("int too small for size")

        return data

    def decode(
        self, data: int | None = None, size_hint: int | None = None
    ) -> PointerOrPointerGenerator[int]:
        if size_hint is None or size_hint < 1:
            size_hint = 8

        ba = bytearray(size_hint)

        yield (ba, size_hint)

        return int.from_bytes(ba, signed=self.signed)
