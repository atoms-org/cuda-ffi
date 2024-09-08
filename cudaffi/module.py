from __future__ import annotations

import warnings

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
# types:
# https://docs.python.org/3/library/ctypes.html#ctypes-fundamental-data-types-2
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


# KernelArgs = CudaData | list[CudaData] | None


class GridSpecCallback(Protocol):
    def __call__(self, name: str, mod: CudaModule, *args: Any) -> GridSpec: ...


class BlockSpecCallback(Protocol):
    def __call__(self, name: str, mod: CudaModule, *args: Any) -> BlockSpec: ...


class CudaFunctionNameNotFound(Exception):
    pass


class CudaFunction:
    def __init__(self, mod: CudaModule, name: str) -> None:
        ret = cuda.cuModuleGetFunction(mod.nv_module, name.encode())
        if ret[0] == cuda.CUresult.CUDA_ERROR_NOT_FOUND:
            raise CudaFunctionNameNotFound(f"CUDA function '{name}' does not exist")

        # check for other errors
        checkCudaErrors(ret)

        self.__nv_kernel__: NvKernel = ret[1]
        self.__cuda_module__ = mod
        self.__name__ = name
        self.__default_grid__: GridSpecCallback | GridSpec = (1, 1, 1)
        self.__default_block__: BlockSpecCallback | BlockSpec = (1, 1, 1)

    def __call__(self, *args: Any, stream: CudaStream | None = None) -> None:
        # name: str,
        # args: KernelArgs = None,
        # *,
        # block: BlockSpec = (1, 1, 1),
        # grid: GridSpec = (1, 1, 1),
        # stream: CudaStream | None = None,

        if stream is None:
            stream = CudaStream.get_default()

        print(f"Calling function: {self.__name__} with args: {args}")  # noqa: T201

        nv_args = CudaFunction._make_args(*args)

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
    def _make_args(*args: Any) -> NvKernelArgs:
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


class CudaCompilationError(Exception):
    def __init__(
        self,
        msg: str,
        compilation_results: str,
        code: str,
        compilation_args: list[bytes],
        mod: CudaModule,
    ) -> None:
        super().__init__(msg)
        self.compilation_results = compilation_results
        self.comilation_args = compilation_args
        self.code = code
        self.module = mod

    def __str__(self) -> str:
        err_str = super().__str__()
        return f"{err_str}\n\nError: CUDA compilation results:\n{self.compilation_results}"


class CudaCompilationWarning(UserWarning):
    def __init__(self, msg: str, compilation_results: str, code: str, mod: CudaModule) -> None:
        super().__init__(msg)
        self.compilation_results = compilation_results
        self.code = code
        self.module = mod

    def __str__(self) -> str:
        err_str = super().__str__()
        return f"{err_str}\n\Warning: CUDA compilation results:\n{self.compilation_results}"


class CudaModule:
    """A CUDA source module"""

    # TODO: include paths, compiler flags
    # https://documen.tician.de/pycuda/driver.html#pycuda.compiler.SourceModule
    # class pycuda.compiler.SourceModule(source, nvcc='nvcc', options=None,
    # keep=False, no_extern_c=False, arch=None, code=None, cache_dir=None,
    # include_dirs=[])
    #
    # arch and code specify the values to be passed for the -arch and -code
    # options on the nvcc command line
    #
    # options=None
    # include_dirs=[]
    def __init__(
        self,
        code: str,
        *,
        progname: str = "<<CudaModule>>",
        compile_options: list[str] | None = None,
        include_dirs: list[str] | None = None,
        no_extern: bool = False,
        device: CudaDevice | None = None,
        stream: CudaStream | None = None,
    ) -> None:
        init()

        if device is None:
            device = CudaDevice.get_default()

        self.progname = progname
        if not no_extern:
            self.code = 'extern "C" {\n' + code + "\n}\n"
        else:
            self.code = code
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
        self.compile_args = CudaModule._make_compile_flags(compile_options, include_dirs)

        compile_result = nvrtc.nvrtcCompileProgram(
            self.nv_prog, len(self.compile_args), self.compile_args
        )

        log_sz = checkCudaErrors(nvrtc.nvrtcGetProgramLogSize(self.nv_prog))
        buf = b" " * log_sz
        checkCudaErrors(nvrtc.nvrtcGetProgramLog(self.nv_prog, buf))
        self.compile_log = buf.decode()

        if log_sz > 1:
            print(f"Compilation results ({log_sz} bytes): {self.compile_log}")  # noqa: T201
        else:
            print("Compilation complete, no warnings.")  # noqa: T201

        # check if the compiler didn't like the arguments
        if compile_result[0] == nvrtc.nvrtcResult.NVRTC_ERROR_INVALID_OPTION:
            raise CudaCompilationError(
                f"Invalid compiler option(s) while compiling code in '{progname}'",
                self.compile_log,
                self.code,
                self.compile_args,
                self,
            )

        # check if compilation failed
        if compile_result[0] == nvrtc.nvrtcResult.NVRTC_ERROR_COMPILATION:
            raise CudaCompilationError(
                f"Error while compiling code in '{progname}'",
                self.compile_log,
                self.code,
                self.compile_args,
                self,
            )

        # check for any compiler output
        if log_sz > 1:
            warnings.warn(
                CudaCompilationWarning(
                    f"Warning while compiling code in '{progname}'",
                    self.compile_log,
                    self.code,
                    self,
                )
            )

        checkCudaErrors(compile_result)

        # Get PTX from compilation
        self.nv_ptx_size = checkCudaErrors(nvrtc.nvrtcGetPTXSize(self.nv_prog))
        self.ptx = b" " * self.nv_ptx_size
        checkCudaErrors(nvrtc.nvrtcGetPTX(self.nv_prog, self.ptx))

        # Load PTX as module data
        self.ptx = np.char.array(self.ptx)
        ret = cuda.cuModuleLoadData(self.ptx.ctypes.data)
        self.nv_module = checkCudaErrors(ret)

    @staticmethod
    def _make_compile_flags(
        opts: list[str] | None,
        include_dirs: list[str] | None,
    ) -> list[bytes]:
        ret: list[bytes] = []

        if opts is not None:
            for opt in opts:
                ret.append(opt.encode())

        if include_dirs is not None:
            for incl in include_dirs:
                ret.append("-I".encode())
                ret.append(incl.encode())

        return ret

    def get_function(self, name: str) -> CudaFunction:
        return CudaFunction(self, name)

    def __getattr__(self, name: str) -> CudaFunction:
        return self.get_function(name)

    # def __del__(self) -> None:
    #     # if compile fails, the nv_module attribute hasn't been set het
    #     if hasattr(self, "nv_module"):
    #         checkCudaErrors(cuda.cuModuleUnload(self.nv_module))

    @staticmethod
    def from_file(filename: str) -> CudaModule:
        with open(filename) as f:
            code = f.read()
        return CudaModule(code=code, progname=filename)
