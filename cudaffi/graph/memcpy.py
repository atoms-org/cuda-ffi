from collections.abc import Buffer

from cuda import cuda, cudart

from ..memory import CudaDeviceMemory, CudaHostMemory, CudaManagedMemory, CudaMemory, HostBuffer
from ..utils import checkCudaErrorsAndReturn
from .graph import CudaGraph, GraphNode


class CudaMemcpyNode(GraphNode):
    def __init__(
        self, g: CudaGraph, src: HostBuffer | CudaMemory, dst: HostBuffer | CudaMemory, size: int
    ) -> None:
        super().__init__(g, "Memcpy")
        self.src = src
        self.dst = dst
        self.size = size
        self.src_type: str
        self.dst_type: str
        self.nv_src: int | Buffer | cudart.cudaHostPtr | cudart.cudaDevPtr
        self.nv_dst: int | Buffer | cudart.cudaHostPtr | cudart.cudaDevPtr

        match src:
            case HostBuffer():
                self.nv_src = src.to_host_nv_data()
                self.src_type = "host"
            case CudaHostMemory():
                self.nv_src = src.dev_addr
                self.src_type = "host"
            case CudaManagedMemory():
                self.nv_src = src.dev_addr
                self.src_type = "managed"
            case CudaDeviceMemory():
                self.nv_src = src.dev_addr
                self.src_type = "device"
            case _:
                raise Exception("unknown src memory type in CudaMemcpyNode")

        match dst:
            case HostBuffer():
                self.nv_dst = dst.to_host_nv_data()
                self.dst_type = "host"
            case CudaHostMemory():
                self.nv_dst = dst.dev_addr
                self.dst_type = "host"
            case CudaManagedMemory():
                self.nv_dst = dst.dev_addr
                self.dst_type = "managed"
            case CudaDeviceMemory():
                self.nv_dst = dst.dev_addr
                self.dst_type = "device"
            case _:
                raise Exception("unknown src memory type in CudaMemcpyNode")

        self.nv_memcpy_node: cuda.CUgraphNode | None = None

        self.nv_node = checkCudaErrorsAndReturn(
            cudart.cudaGraphAddMemcpyNode1D(
                self.graph.nv_graph,
                None,
                0,
                self.nv_dst,
                self.nv_src,
                self.size,
                cudart.cudaMemcpyKind.cudaMemcpyDefault,
            )
        )
