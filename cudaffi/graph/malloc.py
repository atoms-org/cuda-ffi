from typing import NewType, cast

from cuda import cudart

from ..utils import checkCudaErrors
from .graph import CudaGraph, GraphNode

NvMallocNode = NewType("NvMallocNode", object)
CudaMemAddr = NewType("CudaMemAddr", int)


class CudaMallocNode(GraphNode):
    def __init__(self, g: CudaGraph, size: int) -> None:
        super().__init__(g, "Malloc")
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
        nv_memalloc_params.poolProps.allocType = (
            cudart.cudaMemAllocationType.cudaMemAllocationTypePinned
        )
        nv_memalloc_params.poolProps.location.type = (
            cudart.cudaMemLocationType.cudaMemLocationTypeDevice
        )
        # nv_memalloc_params.poolProps = cudart.cudaMemAllocationType.cudaMemAllocationTypePinned
        # nv_memalloc_params.accessDescs
        # nv_memalloc_params.accessDescCount
        # nv_memalloc_params.dptr
        # nv_memalloc_params.getPtr()
        self.nv_memalloc_params = nv_memalloc_params
        print("getPtr", hex(nv_memalloc_params.getPtr()))
        self.nv_node = checkCudaErrors(
            cudart.cudaGraphAddMemAllocNode(self.graph.nv_graph, None, 0, self.nv_memalloc_params)
        )
        print("getPtr", hex(nv_memalloc_params.getPtr()))

    @property
    def addr(self) -> CudaMemAddr:
        return cast(CudaMemAddr, self.nv_memalloc_params.getPtr())
