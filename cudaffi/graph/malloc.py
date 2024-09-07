from typing import NewType

from cuda import cudart

from ..utils import checkCudaErrors
from .graph import CudaGraph, GraphNode

NvMallocNode = NewType("NvMallocNode", object)


class MallocNode(GraphNode):
    def __init__(self, size: int) -> None:
        self.size = size
        self.nv_memory: NvMallocNode | None = None

        nv_memalloc_params = cudart.cudaMemAllocNodeParams()
        # ∕∕ parameters for a basic allocation cuda
        # MemAllocNodeParams params = {};
        # params.poolProps.allocType = cudaMemAllocationTypePinned;
        # params.poolProps.location.type = cudaMemLocationTypeDevice;
        # ∕∕ specify device 0 as the resident device
        # params.poolProps.location.id = 0;
        # params.bytesize = size
        nv_memalloc_params.bytesize = size
        # nv_memalloc_params.poolProps
        # nv_memalloc_params.accessDescs
        # nv_memalloc_params.accessDescCount
        # nv_memalloc_params.dptr
        # nv_memalloc_params.getPtr()
        self.nv_memalloc_params = nv_memalloc_params

    def __nv_mknode__(self, graph: CudaGraph) -> None:
        self.nv_memory = checkCudaErrors(
            cudart.cudaGraphAddMemAllocNode(graph.nv_graph, None, 0, self.nv_memalloc_params)
        )
