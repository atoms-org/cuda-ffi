# @staticmethod
# def from_np(arr: numpy.ndarray, *, stream: CudaStream | None = None) -> CudaMemory:
#     if stream is None:
#         dev = CudaDevice.default()
#         stream = dev.default_stream

#     num_bytes = len(arr) * arr.itemsize
#     mem = CudaMemory(num_bytes)
#     # print("mem.nv_memory", mem.nv_memory)
#     # print("arr.ctypes.data", arr.ctypes.data)
#     # print("num_bytes", num_bytes)
#     # print("stream", stream)
#     checkCudaErrors(
#         cuda.cuMemcpyHtoDAsync(mem.nv_memory, arr.ctypes.data, num_bytes, stream.nv_stream)
#     )

#     return mem
