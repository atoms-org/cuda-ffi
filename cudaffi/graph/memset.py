from typing import NewType

import numpy as np
from cuda import cudart

from ..memory import CudaMemory
from ..utils import checkCudaErrors
from .graph import CudaGraph, GraphNode

NvMemsetNode = NewType("NvMemsetNode", object)


class CudaMemsetNode(GraphNode):
    def __init__(self, g: CudaGraph, mem: CudaMemory, value: int, size: int) -> None:
        super().__init__(g, "Memset")
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

        self.nv_node = checkCudaErrors(
            cudart.cudaGraphAddMemsetNode(self.graph.nv_graph, None, 0, self.nv_memset_params)
        )
