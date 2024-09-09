# mod = CudaSourceFile("print_buf.cu")
# str = bytearray(b"hi there.")
# str.append(0)
# arr = np.array(str, dtype=np.uint8)
# mem = CudaMemory.from_np(arr)
# print("mem hex", hex(mem.nv_memory))
# mod.call("print_buf", [CudaData(mem), CudaData(mem.size)])

import ctypes
from typing import Any

from .base import AnyCType, CudaDataType


class CudaStrDataType(CudaDataType):
    def convert(self, data: Any, name: str) -> tuple[int, AnyCType] | None:
        if isinstance(data, str):
            return (0, ctypes.c_uint)

        return None
