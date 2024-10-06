import ctypes
import re
import warnings
from collections.abc import Buffer
from typing import Any, Generator

from ..args import CudaDataType, CudaDataTypeWarning


class CudaIntSizeWarning(CudaDataTypeWarning):
    pass


class CudaIntDataType(CudaDataType[int]):
    default_direction = "input"
    aliases = ["uint", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"]

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.signed = not name.startswith("u")
        self.int_name = name

        # calculate byte size of this flavor of int
        specified_bits_match = re.search(r"\d+$", name)
        if specified_bits_match is None:
            specified_bits = 32
        else:
            specified_bits = int(specified_bits_match.group())
        self.byte_size = specified_bits // 8

        # calculate value range of this flavor of it
        num_bits = self.byte_size * 8
        if self.signed:
            num_bits -= 1
            max_size = (2**num_bits) - 1
            min_size = (2**num_bits) * -1
        else:
            max_size = (2**num_bits) - 1
            min_size = 0

        self.num_bits = num_bits
        self.max_size = max_size
        self.min_size = min_size

    def is_type(self, data: Any) -> bool:
        return isinstance(data, int)

    def get_byte_size(self, data: int) -> int:
        return self.byte_size

    def get_ctype(self, data: int) -> type[ctypes.c_int]:
        return ctypes.c_int

    def encode(self, data: int) -> int:
        if data > self.max_size:
            warnings.warn(
                f"passed int value '{data}' too large for {self.byte_size} bytes of {self.int_name}",
                CudaDataTypeWarning,
            )

        if data < self.min_size:
            warnings.warn(
                f"passed int value '{data}' too small for {self.byte_size} bytes of {self.int_name}",
                CudaDataTypeWarning,
            )

        return data

    def decode(
        self, data: int | None = None, size_hint: int | None = None
    ) -> Generator[tuple[Buffer, int], Any, int]:
        if size_hint is None or size_hint < 1:
            size_hint = 8

        ba = bytearray(size_hint)

        yield (ba, size_hint)

        return int.from_bytes(ba, signed=self.signed)
