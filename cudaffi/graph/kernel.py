from typing import Any, NewType

from cuda import cuda

from ..module import BlockSpec, CudaFunction, GridSpec
from ..utils import checkCudaErrors
from .graph import CudaGraph, GraphNode

NvKernelNode = NewType("NvKernelNode", object)


class CudaKernelNode(GraphNode):
    def __init__(
        self,
        g: CudaGraph,
        fn: CudaFunction,
        *args: Any,
        block: BlockSpec = (1, 1, 1),
        grid: GridSpec = (1, 1, 1),
    ) -> None:
        super().__init__(g, "Kernel")
        self.block = block
        self.grid = grid
        self.fn = fn
        self.args = args

        print("args", args)

        self.nv_args = CudaFunction._make_args(*args)

        print("nv_args", self.nv_args)

        nv_kernel_node_params = cuda.CUDA_KERNEL_NODE_PARAMS()
        nv_kernel_node_params.func = self.fn._nv_kernel
        nv_kernel_node_params.gridDimX = self.grid[0]
        nv_kernel_node_params.gridDimY = self.grid[1]
        nv_kernel_node_params.gridDimZ = self.grid[2]
        nv_kernel_node_params.blockDimX = self.block[0]
        nv_kernel_node_params.blockDimY = self.block[1]
        nv_kernel_node_params.blockDimZ = self.block[2]
        nv_kernel_node_params.sharedMemBytes = 0
        nv_kernel_node_params.kernelParams = self.nv_args
        self.nv_kernel_node_params = nv_kernel_node_params

        self.nv_kernel_node: NvKernelNode | None = None

        self.nv_node = checkCudaErrors(
            cuda.cuGraphAddKernelNode(self.graph.nv_graph, None, 0, self.nv_kernel_node_params)
        )
