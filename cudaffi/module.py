from __future__ import annotations

"""CUDA Source Modules"""

import ctypes
from typing import Any, NewType, Protocol

import numpy as np
from cuda import cuda, nvrtc

from .core import CudaDevice, CudaStream, init
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


class CudaDataException(Exception):
    pass


class CudaData:
    def __init__(self, data: int | CudaMemory, datatype: NvDataType | None = None) -> None:
        match data:
            case int():
                print("data is int")
                self.data: int | NvMemory = data
            case CudaMemory():
                print("data is CudaMemory")
                self.data = data.nv_memory
                datatype = ctypes.c_void_p
            case _:
                raise CudaDataException(f"can't convert data to CudaData: '{data}'")

        if datatype is None:
            datatype = ctypes.c_uint
        self.type = datatype


KernelArgs = CudaData | list[CudaData] | None


class GridSpecCallback(Protocol):
    def __call__(self, name: str, mod: CudaModule, *args: Any) -> GridSpec: ...


class BlockSpecCallback(Protocol):
    def __call__(self, name: str, mod: CudaModule, *args: Any) -> BlockSpec: ...


class CudaFunction:
    def __init__(self, mod: CudaModule, name: str) -> None:
        self.__cuda_module__ = mod
        self.__name__ = name
        self.__default_grid__: GridSpecCallback | GridSpec = (1, 1, 1)
        self.__default_block__: BlockSpecCallback | BlockSpec = (1, 1, 1)
        self.__nv_kernel__: NvKernel = checkCudaErrors(
            cuda.cuModuleGetFunction(mod.nv_module, name.encode())
        )

    def __call__(self, *args: Any, stream: CudaStream | None = None) -> None:
        print("__call__ args", args)
        # name: str,
        # args: KernelArgs = None,
        # *,
        # block: BlockSpec = (1, 1, 1),
        # grid: GridSpec = (1, 1, 1),
        # stream: CudaStream | None = None,

        if stream is None:
            stream = CudaStream.get_default()

        print(f"Calling function: {self.__name__} with args: {args}")  # noqa: T201

        nv_args = CudaFunction.make_args(*args)

        print("nv_args", nv_args)

        if isinstance(self.__default_grid__, tuple):
            grid = self.__default_grid__
        else:
            grid = self.__default_grid__(self.__name__, self.__cuda_module__, *args)

        if isinstance(self.__default_block__, tuple):
            block = self.__default_block__
        else:
            block = self.__default_block__(self.__name__, self.__cuda_module__, *args)

        checkCudaErrors(
            cuda.cuLaunchKernel(
                self.__nv_kernel__,
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

    @staticmethod
    def make_args(*args: Any) -> NvKernelArgs:
        if len(args) == 0:
            return 0

        converted_args: list[CudaData] = []
        for arg in args:
            if isinstance(arg, CudaData):
                converted_args.append(arg)
            else:
                converted_args.append(CudaData(arg))

        nv_data_args = tuple(arg.data for arg in converted_args)
        nv_type_args = tuple(arg.type for arg in converted_args)
        nv_args = (nv_data_args, nv_type_args)
        return nv_args


class CudaModule:
    """A CUDA source module"""

    # TODO: include paths, compiler flags
    def __init__(
        self,
        code: str,
        *,
        no_extern: bool = False,
        progname: str = "<unspecified>",
        device: CudaDevice | None = None,
        stream: CudaStream | None = None,
    ) -> None:
        init()

        if device is None:
            device = CudaDevice.get_default()

        self.progname = progname
        if not no_extern:
            self.code = 'extern "C" {\n' + code + "\n}\n"
        print(f"CODE:\n-------\n{self.code}\n-------\n")  # noqa: T201

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
            print(f"Compilation results ({log_sz} bytes): {self.compile_log}")  # noqa: T201
        else:
            print("Compilation complete, no warnings.")  # noqa: T201
        checkCudaErrors(compile_result)

        # Get PTX from compilation
        self.nv_ptx_size = checkCudaErrors(nvrtc.nvrtcGetPTXSize(self.nv_prog))
        self.ptx = b" " * self.nv_ptx_size
        checkCudaErrors(nvrtc.nvrtcGetPTX(self.nv_prog, self.ptx))

    def get_function(self, name: str) -> CudaFunction:
        init()

        # Load PTX as module data and retrieve function
        self.ptx = np.char.array(self.ptx)
        print("TODO: better checking here")
        self.nv_module = checkCudaErrors(cuda.cuModuleLoadData(self.ptx.ctypes.data))

        print("returning cuda function")
        return CudaFunction(self, name)

    def __getattr__(self, name: str) -> CudaFunction:
        print("__getattr__")
        return self.get_function(name)

    # def __del__(self) -> None:
    #     checkCudaErrors(cuda.cuModuleUnload(self.nv_module))

    @staticmethod
    def from_file(filename: str) -> CudaModule:
        with open(filename) as f:
            code = f.read()
        return CudaModule(code=code, progname=filename)
