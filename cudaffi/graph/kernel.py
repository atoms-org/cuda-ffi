from typing import Any, NewType

from cuda import cuda

from ..args import CudaArgList
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
        block: BlockSpec | None = None,
        grid: GridSpec | None = None,
    ) -> None:
        super().__init__(g, "Kernel")
        self.fn = fn
        self.args = args

        print("args", args)

        arg_list = CudaArgList(args)
        # arg_list = CudaArgList(args, self.arg_types)
        # arg_list.copy_to_device()
        self.nv_args = arg_list.to_nv_args()

        print("nv_args", self.nv_args)

        if grid is None:
            grid = fn.default_grid
        if block is None:
            block = fn.default_block

        self.block = block
        self.grid = grid

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
