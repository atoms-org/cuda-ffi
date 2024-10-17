import ctypes
from collections.abc import Buffer
from typing import Any

from ..args import CudaDataType


class CudaFloatDataType(CudaDataType[float]):
    default_direction = "input"
    aliases = ["float16", "float32", "float64"]

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.float_name = name
        match self.float_name:
            case "float":
                self.num_bits = 32
            case "float32":
                self.num_bits = 32
            case "float16":
                self.num_bits = 16
            case "float64":
                self.num_bits = 64
            case _:
                raise Exception("internal error in CudaFloatDataType")

        self.num_bytes = self.num_bits // 8
        # TODO: max_size / min_size to detect value truncation

    def is_type(self, data: Any) -> bool:
        return isinstance(data, float)

    def get_byte_size(self, data: float) -> int:
        return self.num_bytes

    def get_ctype(self, data: float) -> type[ctypes.c_float]:
        return ctypes.c_float

    def encode(self, data: float) -> float:
        return data

    def decode(self, data: float | None = None, size_hint: int | None = None) -> tuple[Buffer, int]:
        raise Exception("decoding to float not supported")
