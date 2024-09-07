from typing import NewType

import numpy as np
from cuda import cudart

from ..memory import CudaMemory
from ..utils import checkCudaErrors
from .graph import CudaGraph, GraphNode

NvMemsetNode = NewType("NvMemsetNode", object)


class MemsetNode(GraphNode):
    def __init__(self, mem: CudaMemory, value: int, size: int) -> None:
        self.size = size
        self.value = value
        self.nv_memset_node: NvMemsetNode | None = None

        nv_memset_params = cudart.cudaMemsetParams()
        nv_memset_params.dst = mem.nv_memory
        nv_memset_params.value = self.value
        # nv_memset_params.elementSize = np.dtype(np.float32).itemsize
        nv_memset_params.elementSize = np.dtype(np.uint8).itemsize
        nv_memset_params.width = self.size
        nv_memset_params.height = 1
        self.nv_memset_params = nv_memset_params

    def __nv_mknode__(self, graph: CudaGraph) -> None:
        self.nv_memset_node = checkCudaErrors(
            cudart.cudaGraphAddMemsetNode(graph.nv_graph, None, 0, self.nv_memset_params)
        )
