from cuda import cuda

from cudaffi.graph.graph import CudaGraph, GraphNode
from cudaffi.utils import checkCudaErrorsAndReturn

# from cudaffi.graph.malloc import CudaMallocNode

# from cudaffi.graph.malloc import CudaMallocNode, CudaMemAddr
# from cudaffi.graph.memcpy import CudaMemcpyNode
# from cudaffi.graph.memset import CudaMemsetNode
# from cudaffi.memory import CudaMemory


class TGraphNode(GraphNode):
    def __init__(self, g: CudaGraph, name: str) -> None:
        super().__init__(g, name)
        self.nv_node = checkCudaErrorsAndReturn(cuda.cuGraphAddEmptyNode(g.nv_graph, None, 0))


class TestGraph:
    def test_exists(self) -> None:
        CudaGraph()

    def test_to_networkx(self) -> None:
        g = CudaGraph()
        n1 = TGraphNode(g, "Bob")
        n2 = TGraphNode(g, "Sam")
        n1.depends_on(n2)

        nxg = g.to_networkx()
        nodes = list(nxg.nodes)
        assert len(nodes) == 2
        edges = list(nxg.edges)
        assert len(edges) == 1
        assert edges == [(id(n2), id(n1))]


class TestGraphNode:
    def test_dep(self) -> None:
        g = CudaGraph()
        n1 = TGraphNode(g, "Bob")
        n2 = TGraphNode(g, "Sam")
        n1.depends_on(n2)

        assert len(g.nodes) == 2
        assert n1.before == [n2]
        assert n2.after == [n1]
        assert g.roots == [n2]


# class TestKernelNode:
#     def test_basic(self) -> None:
#         mod = CudaModule.from_file("tests/helpers/simple.cu")
#         fn = mod.get_function("simple")
#         g = CudaGraph()
#         CudaKernelNode(g, fn)
#         g.run()


# class TestMemsetNode:
#     def test_basic(self) -> None:
#         mod = CudaModule.from_file("tests/helpers/print_buf.cu")
#         fn = mod.get_function("print_buf")
#         arr = np.array([1, 2, 3, 4], dtype=np.uint8)
#         num_bytes = len(arr) * arr.itemsize
#         print("num_bytes is", num_bytes)
#         mem = CudaMemory(num_bytes)
#         g = CudaGraph()
#         ms = CudaMemsetNode(g, mem, 42, num_bytes)
#         ker = CudaKernelNode(g, fn, mem, mem.size)
#         ker.depends_on(ms)
#         g.run()


# class TestMemcpyNode:
#     def test_basic(self) -> None:
#         mod = CudaModule.from_file("tests/helpers/print_buf.cu")
#         fn = mod.get_function("print_buf")
#         g = CudaGraph()

#         # mod = CudaSourceFile("print_buf.cu")
#         # str = bytearray(b"hi there.")
#         # str.append(0)
#         # arr = np.array(str, dtype=np.uint8)
#         # mem = CudaMemory.from_np(arr)
#         # print("mem hex", hex(mem.nv_memory))
#         # mod.call("print_buf", [CudaData(mem), CudaData(mem.size)])
#         arr = np.array([1, 2, 3, 4], dtype=np.uint32)
#         mem = CudaMemory(4)
#         mem2 = CudaMemory(4)
#         # try:
#         CudaMemcpyNode(g, mem2, mem, mem.size, "host_to_device")
#         # except Exception as e:
#         #     print("e", e)
#         g.run()


# class TestMallocNode:
#     def test_exists(self) -> None:
#         g = CudaGraph()
#         CudaMallocNode(g, 1)

#     # def test_does_malloc(self) -> None:
#     #     g = CudaGraph()
#     #     mem = CudaMallocNode(g, 4)
#     #     arr = np.array([1, 2, 3, 4], dtype=np.uint32)
#     #     CudaMemcpyNode(g, cast(CudaMemAddr, arr.ctypes.data), mem.addr, 4, "host_to_device")
#     #     g.run()
