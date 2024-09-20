import ctypes
from collections.abc import Buffer
from typing import Any

from ..memory import AnyCType, CudaDataType, PointerOrPointerGenerator


class CudaStrDataType(CudaDataType[str]):
    def is_type(self, data: Any) -> bool:
        return isinstance(data, str)

    def get_byte_size(self, data: str) -> int:
        return len(data) + 1

    def get_ctype(self, data: str) -> AnyCType:
        return ctypes.c_void_p

    def encode(self, data: str) -> tuple[Buffer, int]:
        s = bytearray(data.encode())
        s.append(0)

        print("str encode returning", s)
        return (s, len(data) + 1)

    def decode(
        self, data: str | None = None, size_hint: int | None = None
    ) -> PointerOrPointerGenerator[str]:
        if size_hint is None or size_hint < 1:
            size_hint = 4096

        ba = bytearray(size_hint)

        ret = yield (ba, size_hint)

        s = str(ba)

        return s

    # def to_device(self, src: str, dst: CudaDeviceMemory) -> None:
    #     s = bytearray(src.encode())
    #     s.append(0)
    #     checkCudaErrors(cuda.cuMemcpyHtoD(dst.nv_device_memory, s, len(s)))

    # def to_host(self, src: CudaDeviceMemory, dst: str) -> None:
    #     checkCudaErrors(cuda.cuMemcpyDtoH(dst, src.nv_device_memory, len(dst)))

    #     return None
