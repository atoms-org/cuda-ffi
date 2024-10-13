from __future__ import annotations

from abc import ABC

from cuda import cuda

from ..device import CudaDevice, CudaStream, init
from ..utils import checkCudaErrorsAndReturn, checkCudaErrorsNoReturn


class GraphNode(ABC):
    def __init__(self, graph: CudaGraph, type_name: str) -> None:
        self.name = type_name
        self.graph = graph
        self.nv_node: cuda.CUgraphNode | None = None
        self.graph.nodes.append(self)

    def depends_on(self, n: GraphNode) -> None:
        assert self.nv_node is not None
        assert n.nv_node is not None
        checkCudaErrorsNoReturn(
            cuda.cuGraphAddDependencies(self.graph.nv_graph, [n.nv_node], [self.nv_node], 1)
        )


class CudaGraph:
    # https://github.com/NVIDIA/cuda-python/blob/main/examples/3_CUDA_Features/simpleCudaGraphs_test.py
    def __init__(self, *, stream: CudaStream | None = None) -> None:
        init()

        if not CudaGraph.is_supported():
            raise Exception("current device compute capability does not support CUDA graphs")

        self.nodes: list[GraphNode] = []
        self.nv_graph: cuda.CUgraph = checkCudaErrorsAndReturn(cuda.cuGraphCreate(0))

    def run(self, stream: CudaStream | None = None) -> None:
        print("*** RUNNING CUDAGRAPH ***")
        self.nv_graph_exec = checkCudaErrorsAndReturn(cuda.cuGraphInstantiate(self.nv_graph, 0))

        if stream is None:
            stream = CudaStream.get_default()

        checkCudaErrorsNoReturn(cuda.cuGraphLaunch(self.nv_graph_exec, stream.nv_stream))

        stream.synchronize()

    @classmethod
    def is_supported(self, dev: CudaDevice | None = None) -> bool:
        if dev is None:
            dev = CudaDevice.get_default()

        # int driverVersion = 0;
        # int deviceSupportsMemoryPools = 0;
        # cudaDriverGetVersion(&driverVersion);
        # if (driverVersion >= 11020) { ∕∕ avoid invalid value error in cudaDeviceGetAttribute
        #     cudaDeviceGetAttribute(&deviceSupportsMemoryPools, cudaDevAttrMemoryPoolsSupported, device);
        # }
        # deviceSupportsMemoryNodes = (driverVersion >= 11040) && (deviceSupportsMemoryPools != 0);

        if dev.get_attribute("memory_pools_supported") != 1:
            return False

        major, minor = dev.driver_version
        if major > 11:
            return True
        if major == 11 and minor >= 4:
            return True

        return False

    @property
    def nv_nodes(self, max: int = 1024) -> list[cuda.CUgraphNode]:
        nodes, num_nodes = checkCudaErrorsAndReturn(cuda.cuGraphGetNodes(self.nv_graph, max))
        return nodes[:num_nodes]

    @property
    def nv_root_nodes(self, max: int = 1024) -> list[cuda.CUgraphNode]:
        nodes, num_nodes = checkCudaErrorsAndReturn(cuda.cuGraphGetRootNodes(self.nv_graph, max))
        return nodes[:num_nodes]

    @property
    def nv_edges(self, max: int = 1024) -> list[tuple[cuda.CUgraphNode, cuda.CUgraphNode]]:
        ret: list[tuple[cuda.CUgraphNode, cuda.CUgraphNode]] = []

        from_nodes, to_nodes, num_nodes = checkCudaErrorsAndReturn(
            cuda.cuGraphGetEdges(self.nv_graph, max)
        )

        for n in range(num_nodes):
            ret.append((from_nodes[n], to_nodes[n]))

        return ret

    ######################
    # Future expansions?
    ######################
    # cuStreamBeginCaptureToGraph
    # instantiate()
    # upload()
    # launch()

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

    # root_nodes[]
    # edges[]
    # to_dot()
    # to_networkx()
