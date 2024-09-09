from enum import Enum
from typing import NewType

from cuda import cudart

from ..memory import CudaMemory, NvMemory
from ..utils import checkCudaErrors
from .graph import CudaGraph, GraphNode
from .malloc import CudaMemAddr

NvMemcpyNode = NewType("NvMemcpyNode", object)


class CopyDirection(Enum):
    device_to_host = 1
    host_to_device = 2


class CudaMemcpyNode(GraphNode):
    def __init__(
        self,
        g: CudaGraph,
        src: CudaMemory | CudaMemAddr,
        dst: CudaMemory | CudaMemAddr,
        size: int,
        direction: str,
    ) -> None:
        super().__init__(g, "Memcpy")
        self.src = src
        self.dst = dst
        self.size = size
        self.direction = CopyDirection[direction]

        if isinstance(self.src, CudaMemory):
            self.nv_src: CudaMemAddr | NvMemory = self.src.nv_memory
        else:
            self.nv_src = self.src

        if isinstance(self.dst, CudaMemory):
            self.nv_dst: CudaMemAddr | NvMemory = self.dst.nv_memory
        else:
            self.nv_dst = self.dst

        self.nv_memcpy_node: NvMemcpyNode | None = None

        nv_memcpy_params = cudart.cudaMemcpy3DParms()
        # nv_memcpy_params.srcArray = None
        # nv_memcpy_params.srcPos = cudart.make_cudaPos(0, 0, 0)
        # nv_memcpy_params.srcPtr = cudart.make_cudaPitchedPtr(
        #     self.nv_src, np.dtype(np.int32).itemsize, 1, 1
        # )
        # nv_memcpy_params.dstArray = None
        # nv_memcpy_params.dstPos = cudart.make_cudaPos(0, 0, 0)
        # nv_memcpy_params.dstPtr = cudart.make_cudaPitchedPtr(
        #     self.nv_dst, np.dtype(np.int32).itemsize, 1, 1
        # )
        # nv_memcpy_params.srcPtr = self.nv_src
        # nv_memcpy_params.dstPtr = self.nv_dst
        print("self.nv_src", self.nv_src)
        print("self.nv_dst", self.nv_dst)
        src_pp = cudart.make_cudaPitchedPtr(self.nv_src, 1, self.size, 1)
        dst_pp = cudart.make_cudaPitchedPtr(self.nv_dst, 1, self.size, 1)
        # nv_memcpy_params.extent = cudart.make_cudaExtent(np.dtype(np.float64).itemsize, 1, 1)
        if self.direction == CopyDirection.device_to_host:
            nv_memcpy_params.kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
            self.direction = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        else:
            nv_memcpy_params.kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
            self.direction = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        # self.nv_memcpy_params = nv_memcpy_params

        # self.nv_node = checkCudaErrors(
        #     cudart.cudaGraphAddMemcpyNode(self.graph.nv_graph, None, 0, self.nv_memcpy_params)
        # )

        foo = checkCudaErrors(cudart.cudaMallocHost(self.size))
        self.nv_node = checkCudaErrors(
            cudart.cudaGraphAddMemcpyNode1D(
                self.graph.nv_graph,
                None,
                0,
                bytearray([1, 2, 3, 4]),
                foo,
                self.size,
                cudart.cudaMemcpyKind.cudaMemcpyHostToHost,
            )
        )
        self.nv_node = checkCudaErrors(
            cudart.cudaGraphAddMemcpyNode1D(
                self.graph.nv_graph,
                None,
                0,
                foo,
                self.nv_dst,
                self.size,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
        )
