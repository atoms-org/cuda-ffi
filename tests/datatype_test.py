from typing import Any

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
