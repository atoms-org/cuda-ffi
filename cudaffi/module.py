from __future__ import annotations

"""CUDA Source Modules"""

import ctypes
from typing import Any, NewType

import numpy as np
from cuda import cuda, nvrtc

from .core import CudaDevice, CudaStream
from .memory import CudaMemory, NvMemory
from .utils import checkCudaErrors

NvProgram = NewType("NvProgram", object)  # nvrtc.nvrtcGetProgram
BlockSpec = tuple[int, int, int]
GridSpec = tuple[int, int, int]


NvKernel = NewType("NvKernel", object)  # cuda.CUkernel
NvDataType = type[ctypes.c_uint] | type[ctypes.c_void_p]
NvKernelArgs = (
    int  # for None args
    | tuple[
        tuple[Any, ...],  # list of data
        tuple[Any, ...],  # list of data types
    ]
)


def make_args(args: KernelArgs) -> NvKernelArgs:
    if args is None:
        nv_args: NvKernelArgs | int = 0
    else:
        if isinstance(args, CudaData):
            args = [args]

        nv_data_args = tuple(arg.data for arg in args)
        nv_type_args = tuple(arg.type for arg in args)
        nv_args = (nv_data_args, nv_type_args)
    return nv_args


class CudaData:
    def __init__(self, data: int | CudaMemory, datatype: NvDataType | None = None) -> None:
        if isinstance(data, int):
            self.data: int | NvMemory = data
        if isinstance(data, CudaMemory):
            self.data = data.nv_memory
            datatype = ctypes.c_void_p

        if datatype is None:
            datatype = ctypes.c_uint
        self.type = datatype


KernelArgs = CudaData | list[CudaData] | None


class CudaFunction:
    def __init__(self, src: CudaModule, name: str) -> None:
        self.src = src
        self.name = name
        self.nv_kernel: NvKernel = checkCudaErrors(
            cuda.cuModuleGetFunction(src.nv_module, name.encode())
        )


class CudaModule:
    """A CUDA source module"""

    # TODO: include paths, compiler flags
    def __init__(
        self, code: str, *, no_extern: bool = False, progname: str = "<unspecified>"
    ) -> None:
        device = CudaDevice.default()
        context = device.default_context  # cuModuleLoadData requires a context
        stream = device.default_stream
        self.progname = progname
        if not no_extern:
            self.code = 'extern "C" {\n' + code + "\n}\n"
        print(f"CODE:\n-------\n{self.code}\n-------\n")

        # Create program
        self.nv_prog: NvProgram = checkCudaErrors(
            nvrtc.nvrtcCreateProgram(self.code.encode(), self.progname.encode(), 0, [], [])
        )

        # Compile code
        # Compile program
        # arch_arg = bytes(f"--gpu-architecture=compute_{major}{minor}", "ascii")
        # opts = [b"--fmad=false", arch_arg]
        # ret = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        compile_result = nvrtc.nvrtcCompileProgram(self.nv_prog, 0, [])
        log_sz = checkCudaErrors(nvrtc.nvrtcGetProgramLogSize(self.nv_prog))
        buf = b" " * log_sz
        checkCudaErrors(nvrtc.nvrtcGetProgramLog(self.nv_prog, buf))
        self.compile_log = buf.decode()
        if log_sz > 1:
            print(f"Compilation results ({log_sz} bytes):\b{self.compile_log}")
        else:
            print("Compilation complete, no warnings.")
        checkCudaErrors(compile_result)

        # Get PTX from compilation
        self.nv_ptx_size = checkCudaErrors(nvrtc.nvrtcGetPTXSize(self.nv_prog))
        self.ptx = b" " * self.nv_ptx_size
        checkCudaErrors(nvrtc.nvrtcGetPTX(self.nv_prog, self.ptx))

    def get_function(
        self,
        name: str,
        *,
        device: CudaDevice | None = None,
    ) -> CudaFunction:
        if device is None:
            device = CudaDevice.default()
        context = device.default_context  # cuModuleLoadData requires a context
        stream = device.default_stream

        # Load PTX as module data and retrieve function
        self.ptx = np.char.array(self.ptx)
        self.nv_module = checkCudaErrors(cuda.cuModuleLoadData(self.ptx.ctypes.data))

        return CudaFunction(self, name)

    def call(
        self,
        name: str,
        args: KernelArgs = None,
        *,
        block: BlockSpec = (1, 1, 1),
        grid: GridSpec = (1, 1, 1),
        stream: CudaStream | None = None,
    ) -> None:
        if stream is None:
            device = CudaDevice.default()
            stream = device.default_stream

        print("Calling function:", name)
        fn = self.get_function(name, device=device)

        nv_args = make_args(args)

        print("nv_args", nv_args)

        checkCudaErrors(
            cuda.cuLaunchKernel(
                fn.nv_kernel,
                grid[0],  # grid x dim
                grid[1],  # grid y dim
                grid[2],  # grid z dim
                block[0],  # block x dim
                block[1],  # block y dim
                block[2],  # block z dim
                0,  # dynamic shared memory
                stream.nv_stream,  # stream
                #    args.ctypes.data,  # kernel arguments
                nv_args,  # kernel arguments
                0,  # extra (ignore)
            )
        )

    # def __del__(self) -> None:
    #     checkCudaErrors(cuda.cuModuleUnload(self.nv_module))

    @staticmethod
    def from_file(filename: str) -> CudaModule:
        with open(filename) as f:
            code = f.read()
        return CudaModule(code=code, progname=filename)
