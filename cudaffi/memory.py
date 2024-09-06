from __future__ import annotations

from typing import Any, NewType

import numpy as np
from cuda import cuda, cudart

from .core import CudaContext, CudaDevice, CudaStream
from .utils import checkCudaErrors

NvMemory = NewType("NvMemory", object)  # cuda.CUdeviceptr


class CudaMemory:
    def __init__(self, size: int, ctx: CudaContext | None = None) -> None:
        if ctx is None:
            device = CudaDevice.default()
            ctx = device.default_context

        self.size = size
        self.nv_memory: NvMemory = checkCudaErrors(cudart.cudaMalloc(size))
        # self.nv_memory: NvMemory = checkCudaErrors(cuda.cuMemAlloc(size))

    # def __del__(self) -> None:
    #     checkCudaErrors(cuda.cuMemFree(self.nv_memory))

    @staticmethod
    def from_np(arr: np.ndarray[Any, Any], *, stream: CudaStream | None = None) -> CudaMemory:
        if stream is None:
            dev = CudaDevice.default()
            stream = dev.default_stream

        num_bytes = len(arr) * arr.itemsize
        mem = CudaMemory(num_bytes)
        # print("mem.nv_memory", mem.nv_memory)
        # print("arr.ctypes.data", arr.ctypes.data)
        # print("num_bytes", num_bytes)
        # print("stream", stream)
        checkCudaErrors(
            cuda.cuMemcpyHtoDAsync(mem.nv_memory, arr.ctypes.data, num_bytes, stream.nv_stream)
        )

        return mem

    # cuda.cuMemcpy
    # cuda.cuMemcpyHtoD
    # cuda.cuMemcpyDtoH

    # managed
    # pagelocked
    pass
    # malloc
    # to_device
    # from_device
    # free
    # as_buffer
