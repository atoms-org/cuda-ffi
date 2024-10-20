from __future__ import annotations

from typing import TYPE_CHECKING

from cuda import cuda

from ..utils import checkCudaErrorsAndReturn
from .graph import CudaGraph, GraphNode

if TYPE_CHECKING:
    from ..module import BlockSpec, CudaFunction, GridSpec


class CudaKernelNode(GraphNode):
    def __init__(
        self,
        g: CudaGraph,
        fn: CudaFunction,
        nv_args: cuda.cudaKernelParams,
        dependencies: list[GraphNode] = list(),
        block: BlockSpec | None = None,
        grid: GridSpec | None = None,
    ) -> None:
        super().__init__(g, "Kernel")
        self.fn = fn
        self.nv_args = nv_args

        if grid is None:
            grid = fn.default_grid
        if block is None:
            block = fn.default_block

        self.block = block
        self.grid = grid

        deps: list[cuda.CUgraphNode] | None = None
        deps_len = 0
        if len(dependencies) > 0:
            deps = [n.nv_node for n in dependencies if n.nv_node is not None]
            deps_len = len(deps)

        self.nv_node_params: cuda.CUDA_KERNEL_NODE_PARAMS = cuda.CUDA_KERNEL_NODE_PARAMS()
        self.nv_node_params.func = self.fn._nv_kernel
        self.nv_node_params.gridDimX = grid[0]
        self.nv_node_params.gridDimY = grid[1]
        self.nv_node_params.gridDimZ = grid[2]
        self.nv_node_params.blockDimX = block[0]
        self.nv_node_params.blockDimY = block[1]
        self.nv_node_params.blockDimZ = block[2]
        self.nv_node_params.sharedMemBytes = 0
        self.nv_node_params.kernelParams = self.nv_args

        self.nv_node = checkCudaErrorsAndReturn(
            cuda.cuGraphAddKernelNode(self.graph.nv_graph, deps, deps_len, self.nv_node_params)
        )
