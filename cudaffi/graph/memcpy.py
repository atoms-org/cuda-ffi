from enum import Enum
from typing import NewType

import numpy as np
from cuda import cudart

from ..memory import CudaMemory
from ..utils import checkCudaErrors
from .graph import CudaGraph, GraphNode

NvMemcpyNode = NewType("NvMemcpyNode", object)


class CopyDirection(Enum):
    device_to_host = 1
    host_to_device = 2


class MemcpyNode(GraphNode):
    def __init__(self, src: CudaMemory, dst: CudaMemory, size: int, direction: str) -> None:
        self.src = src
        self.dst = dst
        self.size = size
        self.direction = CopyDirection[direction]
        self.nv_src = self.src.nv_memory
        self.nv_dst = self.dst.nv_memory
        self.nv_memcpy_node: NvMemcpyNode | None = None

        nv_memcpy_params = cudart.cudaMemcpy3DParms()
        nv_memcpy_params.srcArray = None
        nv_memcpy_params.srcPos = cudart.make_cudaPos(0, 0, 0)
        nv_memcpy_params.srcPtr = cudart.make_cudaPitchedPtr(
            self.nv_src, np.dtype(np.float64).itemsize, 1, 1
        )
        nv_memcpy_params.dstArray = None
        nv_memcpy_params.dstPos = cudart.make_cudaPos(0, 0, 0)
        nv_memcpy_params.dstPtr = cudart.make_cudaPitchedPtr(
            self.nv_dst, np.dtype(np.float64).itemsize, 1, 1
        )
        nv_memcpy_params.extent = cudart.make_cudaExtent(np.dtype(np.float64).itemsize, 1, 1)
        if self.direction == CopyDirection.device_to_host:
            nv_memcpy_params.kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        else:
            nv_memcpy_params.kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        self.nv_memcpy_params = nv_memcpy_params

    def __nv_mknode__(self, graph: CudaGraph) -> None:
        self.nv_memcpy_node = checkCudaErrors(
            cudart.cudaGraphAddMemcpyNode(graph.nv_graph, None, 0, self.nv_memcpy_params)
        )
