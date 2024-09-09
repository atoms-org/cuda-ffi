from __future__ import annotations

from abc import ABC
from typing import NewType

from cuda import cuda, cudart

from ..core import CudaStream, init
from ..utils import checkCudaErrors

NvGraphExec = NewType("NvGraphExec", object)  # cuda.CUgraphExec
NvGraph = NewType("NvGraph", object)  # cuda.CUgraph
NvGraphNode = NewType("NvGraphNode", object)  # cuda.CUgraphNode


class GraphNode(ABC):
    def __init__(self, graph: CudaGraph, type_name: str) -> None:
        self.name = type_name
        self.graph = graph
        self.nv_node: NvGraphNode | None = None

    def depends_on(self, n: GraphNode) -> None:
        assert self.nv_node is not None
        assert n.nv_node is not None
        print("self.nv_node", self.nv_node)
        print("n.nv_node", n.nv_node)
        checkCudaErrors(
            cudart.cudaGraphAddDependencies(self.graph.nv_graph, [n.nv_node], [self.nv_node], 1)
        )
        # cudart.cudaGraphAddDependencies(self.graph.nv_graph, [self.nv_node], [n.nv_node], 1)


class CudaGraph:
    # https://github.com/NVIDIA/cuda-python/blob/main/examples/3_CUDA_Features/simpleCudaGraphs_test.py
    def __init__(self, *, stream: CudaStream | None = None) -> None:
        init()

        if stream is None:
            stream = CudaStream.get_default()

        self.stream = stream
        self.nodes: list[GraphNode] = []
        self.nv_graph: NvGraph = checkCudaErrors(cuda.cuGraphCreate(0))

    def run(self) -> None:
        self.nv_graph_exec: NvGraphExec = checkCudaErrors(
            cudart.cudaGraphInstantiate(self.nv_graph, 0)
        )

        checkCudaErrors(cudart.cudaGraphLaunch(self.nv_graph_exec, self.stream.nv_stream))

        self.stream.synchronize()

    def is_supported(self) -> bool:
        # int driverVersion = 0;
        # int deviceSupportsMemoryPools = 0;
        # cudaDriverGetVersion(&driverVersion);
        # if (driverVersion >= 11020) { ∕∕ avoid invalid value error in cudaDeviceGetAttribute
        #     cudaDeviceGetAttribute(&deviceSupportsMemoryPools, cudaDevAttrMemoryPoolsSupported, device);
        # }
        # deviceSupportsMemoryNodes = (driverVersion >= 11040) && (deviceSupportsMemoryPools != 0);
        return False

    # cuStreamBeginCaptureToGraph
    # instantiate()
    # upload()
    # launch()

    # def add_kernel_node(self, fn: CudaFunction) -> None:
    #     kernelNodeParams = cuda.CUDA_KERNEL_NODE_PARAMS()
    #     self.fn = fn
    #     kernelNodeParams.func = fn.kernel  # type: ignore
    #     kernelNodeParams.gridDimX = 1
    #     kernelNodeParams.gridDimY = kernelNodeParams.gridDimZ = 1
    #     kernelNodeParams.blockDimX = 1
    #     kernelNodeParams.blockDimY = kernelNodeParams.blockDimZ = 1
    #     kernelNodeParams.sharedMemBytes = 0
    #     # kernelNodeParams.kernelParams = kernelArgs
    #     kernelNodeParams.kernelParams = 0
    #     checkCudaErrors(cuda.cuGraphAddKernelNode(self.nv_graph, None, 0, kernelNodeParams))

    # add_memcpy_node()
    # add_memset_node()
    # add_host_node()
    # add_child_graph_node()
    # add_empty_node()
    # add_event_record_node()
    # add_event_wait_node()
    # add_external_semaphore_node()
    # add_external_semaphore_wait_node()
    # add_batch_memop_node()
    # add_mem_alloc_node()
    # add_mem_free_node()
    # mem_trim()
    # clone()
    @property
    def nv_nodes(self) -> list[NvGraphNode]:
        nodes: list[NvGraphNode]
        numNodes: int
        nodes, numNodes = checkCudaErrors(cudart.cudaGraphGetNodes(self.nv_graph))
        return nodes

    # root_nodes[]
    # edges[]
    # to_dot()
    # to_networkx()
