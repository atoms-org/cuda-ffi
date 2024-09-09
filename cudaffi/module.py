from __future__ import annotations

import warnings
from enum import Enum

"""CUDA Source Modules"""

import ctypes
from typing import Any, NewType, Protocol

import numpy as np
from cuda import cuda, nvrtc

from .core import CudaDevice, CudaStream, init
from .datatypes import CudaDataType
from .memory import CudaMemory
from .utils import checkCudaErrors

NvProgram = NewType("NvProgram", object)  # nvrtc.nvrtcGetProgram
BlockSpec = tuple[int, int, int]
GridSpec = tuple[int, int, int]


NvKernel = NewType("NvKernel", object)  # cuda.CUkernel
# types:
# https://docs.python.org/3/library/ctypes.html#ctypes-fundamental-data-types-2
NvKernelArgs = (
    int  # for None args
    | tuple[
        tuple[Any, ...],  # list of data
        tuple[Any, ...],  # list of data types
    ]
)


class CudaDataException(Exception):
    pass


NvDataType = type[ctypes.c_uint] | type[ctypes.c_void_p]


class CudaData:
    def __init__(self, data: int | CudaMemory, type: str | None = None) -> None:
        self.data: int
        self.ctype: NvDataType
        self.type: str

        data_type_registry = CudaDataType.get_registry()
        if type is not None:
            if type not in data_type_registry:
                raise Exception(f"'{type}' is not a registered data type")  # TODO
            datatype = data_type_registry[type]

            ret = datatype.convert(data, type)

        for type in data_type_registry:
            ret = data_type_registry[type].convert(data, type)
            if ret is not None:
                break

        if ret is None:
            raise Exception(f"data could not be converted to {type}")  # TODO

        d, ct = ret
        self.type = type
        self.data = d
        self.orig_data = data
        self.ctype = ct
        self.datatype = data_type_registry[type]

        # TODO: allocate memory, direction, etc.

        # self.data: int | str | NvMemory
        # self.type: NvDataType
        # match data:
        #     case int():
        #         print("data is int")
        #         self.data = data
        #         self.type = ctypes.c_uint
        #     case str():
        #         self.data = data
        #         self.type = ctypes.c_void_p
        #     case CudaMemory():
        #         print("data is CudaMemory")
        #         self.data = data.nv_memory
        #         self.type = ctypes.c_void_p
        #     case _:
        #         raise CudaDataException(f"can't convert data to CudaData: '{data}'")


class CudaArgDirection(Enum):
    input = 1
    output = 2
    inout = 3


class CudaArgType:
    def __init__(
        self,
        name: str | None,
        type: str | None = None,
        direction: str = "inout",
    ) -> None:
        try:
            self.direction = CudaArgDirection[direction]
        except:
            raise Exception(f"Invalid arg direction: '{direction}'")  # TODO

        if type is None:
            raise Exception("Unspecified arg type")  # TODO
        self.type = type


CudaArgTypeList = list[CudaArgType]


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

        self._nv_kernel: NvKernel = ret[1]
        self._cuda_module = mod
        self.name = name
        self.arg_types: CudaArgTypeList | None = None
        self._default_grid_fn: GridSpecCallback | None = None
        self._default_grid: GridSpec = (1, 1, 1)
        self._default_block_fn: BlockSpecCallback | None = None
        self._default_block: BlockSpec = (1, 1, 1)

    def __call__(
        self,
        *args: Any,
        grid: GridSpec | None = None,
        block: BlockSpec | None = None,
        stream: CudaStream | None = None,
    ) -> None:
        if stream is None:
            stream = CudaStream.get_default()

        print(f"Calling function: {self.name} with args: {args}")

        nv_args = CudaFunction._make_args(*args)

        print("nv_args", nv_args)

        if grid is None:
            grid = self.get_default_grid(*args)
        if block is None:
            block = self.get_default_block(*args)

        checkCudaErrors(
            cuda.cuLaunchKernel(
                self._nv_kernel,
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

    def get_default_grid(self, *args: Any) -> GridSpec:
        if self._default_grid_fn is not None:
            return self._default_grid_fn(self.name, self._cuda_module, *args)
        else:
            return self._default_grid

    def get_default_block(self, *args: Any) -> BlockSpec:
        if self._default_block_fn is not None:
            return self._default_block_fn(self.name, self._cuda_module, *args)
        else:
            return self._default_block

    @staticmethod
    def _make_args(*args: Any, argtypes: CudaArgType | None = None) -> NvKernelArgs:
        if len(args) == 0:
            return 0

        converted_args: list[CudaData] = []
        for arg in args:
            if isinstance(arg, CudaData):
                converted_args.append(arg)
            else:
                converted_args.append(CudaData(arg))

        nv_data_args = tuple(arg.data for arg in converted_args)
        nv_type_args = tuple(arg.ctype for arg in converted_args)
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

        # Create program
        self.nv_prog: NvProgram = checkCudaErrors(
            nvrtc.nvrtcCreateProgram(self.code.encode(), self.progname.encode(), 0, [], [])
        )

        # Compile code
        major, minor = device.compute_capability
        co = list() if compile_options is None else compile_options
        co.append(f"--gpu-architecture=compute_{major}{minor}")
        self.compile_args = CudaModule._make_compile_flags(co, include_dirs)

        compile_result = nvrtc.nvrtcCompileProgram(
            self.nv_prog, len(self.compile_args), self.compile_args
        )

        log_sz = checkCudaErrors(nvrtc.nvrtcGetProgramLogSize(self.nv_prog))
        buf = b" " * log_sz
        checkCudaErrors(nvrtc.nvrtcGetProgramLog(self.nv_prog, buf))
        self.compile_log = buf.decode()

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
