import array
from typing import Any

import numpy as np

from cudaffi.datatypes import CudaDataType
from cudaffi.memory import CudaMemory


class CudaTestDataType(CudaDataType):
    def convert(self, type: str, data: Any) -> CudaMemory | None:
        return None


class TestDataType:
    def test_exists(self) -> None:
        CudaTestDataType("test")

    def test_str(self) -> None:
        mem = CudaMemory.from_any("this is a test")
        assert mem.size == 15

    def test_bytes(self) -> None:
        b = bytes([1, 3, 5, 7, 9])
        mem = CudaMemory.from_any(b)
        assert mem.size == 5

    def test_bytearray(self) -> None:
        barr = bytearray([1, 2, 3, 4, 5, 6, 7])
        mem = CudaMemory.from_any(barr)
        assert mem.size == 7

    def test_array(self) -> None:
        arr = array.array("h", [1, 2, 3, 4])
        mem = CudaMemory.from_any(arr)
        assert mem.size == 4 * 2

    def test_numpy(self) -> None:
        arr = np.array([1, 1, 2, 3, 5, 8, 13], dtype=np.int32)
        mem = CudaMemory.from_any(arr)
        assert mem.size == 7 * 4

    def test_numpy_strided(self) -> None:
        arr = np.arange(50, dtype=np.int64)
        arr = arr[::2]
        assert len(arr) == 25
        mem = CudaMemory.from_any(arr)
        assert mem.size == 25 * 8
        # mod = CudaModule.from_file("tests/helpers/print_buf.cu")
        # mod.print_buf(arr, 25 * 8)

    # def test_buffer_protocol(self) -> None:
    #     class BufObj:
    #         def __buffer__(self) -> None:
    #             pass

    #     mem = CudaMemory.from_any(BufObj())
    #     assert mem.size == 7

    # def test_struct(self) -> None:
    #     s = CudaStruct(foo="int32", bar="uint8")
    #     mem = CudaMemory.from_any(s)
    #     assert mem.size == 4 + 1 + 3
