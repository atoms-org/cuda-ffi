"""CUDA Source Modules"""

from __future__ import annotations

import ctypes
import warnings
from typing import Any, Callable, Sequence

import numpy as np
from cuda import cuda, nvrtc

from .args import CudaArgList, CudaArgSpecList, CudaArgType, CudaArgTypeList
from .device import CudaDevice, CudaStream, init
from .graph.graph import CudaGraph
from .graph.kernel import CudaKernelNode
from .utils import (
    checkCudaErrorsAndReturn,
    checkCudaErrorsNoReturn,
)

BlockSpec = tuple[int, int, int]
GridSpec = tuple[int, int, int]


class CudaDataException(Exception):
    pass


NvDataType = type[ctypes.c_uint] | type[ctypes.c_void_p]


class CudaFunctionNameNotFound(Exception):
    pass


class DefaultBlockDescriptor:
    def __set__(self, instance: CudaFunction, val: BlockSpec | BlockSpecCallback) -> None:
        if isinstance(val, tuple):
            instance._default_block = val
        else:
            instance._default_block_fn = val

    def __get__(self, instance: CudaFunction, cls: type[CudaFunction]) -> BlockSpec:
        if instance._default_block_fn is not None:
            return instance._default_block_fn(
                # TODO: not sure why mypy doesn't recognize CudaFunction attrs here
                instance.name,  # type: ignore
                instance._cuda_module,  # type: ignore
                instance._current_args,
            )
        else:
            return instance._default_block


# TODO: block descriptor and grid descriptor could be one generic type
class DefaultGridDescriptor:
    def __set__(self, instance: CudaFunction, val: GridSpec | GridSpecCallback) -> None:
        if isinstance(val, tuple):
            instance._default_grid = val
        else:
            instance._default_grid_fn = val

    def __get__(self, instance: CudaFunction, cls: type[CudaFunction]) -> GridSpec:
        if instance._default_grid_fn is not None:
            return instance._default_grid_fn(
                # TODO: not sure why mypy doesn't recognize CudaFunction attrs here
                instance.name,  # type: ignore
                instance._cuda_module,  # type: ignore
                instance._current_args,
            )
        else:
            return instance._default_grid


class ArgTypeDescriptor:
    def __set__(self, instance: CudaFunction, arg_types: CudaArgSpecList) -> None:
        arg_specs: CudaArgTypeList = list()

        for n in range(len(arg_types)):
            arg_type = arg_types[n]
            if isinstance(arg_type, tuple):
                spec = CudaArgType.from_tuple(arg_type, name=f"arg{n}")
            else:
                spec = CudaArgType(**arg_type)
            arg_specs.append(spec)
        instance._arg_types = arg_specs

    def __get__(self, instance: CudaFunction, cls: type[CudaFunction]) -> CudaArgTypeList | None:
        return instance._arg_types


class CudaFunction:
    default_block = DefaultBlockDescriptor()
    default_grid = DefaultGridDescriptor()
    arg_types = ArgTypeDescriptor()

    def __init__(self, mod: CudaModule, name: str) -> None:
        ret = cuda.cuModuleGetFunction(mod.nv_module, name.encode())

        # if not found, return None
        if ret[0] == cuda.CUresult.CUDA_ERROR_NOT_FOUND:
            raise CudaFunctionNameNotFound(f"CUDA function '{name}' does not exist")

        # raise other errors
        checkCudaErrorsAndReturn(ret)

        self._nv_kernel = ret[1]
        assert isinstance(self._nv_kernel, cuda.CUfunction)
        self._cuda_module = mod
        self.name = name
        self._arg_types: CudaArgTypeList | None = None
        self._default_grid_fn: GridSpecCallback | None = None
        self._default_grid: GridSpec = (1, 1, 1)
        self._default_block_fn: BlockSpecCallback | None = None
        self._default_block: BlockSpec = (1, 1, 1)
        self._current_args: Any = None

    def __call__(
        self,
        *args: Any,
        grid: GridSpec | None = None,
        block: BlockSpec | None = None,
        stream: CudaStream | None = None,
    ) -> Any:
        if stream is None:
            stream = CudaStream.get_default()

        print(f"Calling function: {self.name} with args: {args}")

        arg_list = CudaArgList(args, self.arg_types)
        arg_list.copy_to_device()
        nv_args = arg_list.to_nv_args()

        if grid is None:
            grid = self.default_grid
        if block is None:
            block = self.default_block

        checkCudaErrorsNoReturn(
            cuda.cuLaunchKernel(
                self._nv_kernel,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                0,
                stream.nv_stream,
                nv_args,
                0,
            )
        )

        self._current_args = None

        arg_list.copy_to_host()

        stream.synchronize()

        return arg_list.get_outputs()

    def __repr__(self) -> str:
        return f"{self._cuda_module.progname}:{self.name}:{hex(id(self))}"

    def __str__(self) -> str:
        return f"{self._cuda_module.progname}:{self.name}"


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

        self.fn_cache: dict[str, CudaFunction] = {}

        # Create program
        self.nv_prog = checkCudaErrorsAndReturn(
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

        log_sz = checkCudaErrorsAndReturn(nvrtc.nvrtcGetProgramLogSize(self.nv_prog))
        buf = b" " * log_sz
        checkCudaErrorsNoReturn(nvrtc.nvrtcGetProgramLog(self.nv_prog, buf))
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

        checkCudaErrorsNoReturn(compile_result)

        # Get PTX from compilation
        self.nv_ptx_size = checkCudaErrorsAndReturn(nvrtc.nvrtcGetPTXSize(self.nv_prog))
        self.ptx_buf = b" " * self.nv_ptx_size
        checkCudaErrorsNoReturn(nvrtc.nvrtcGetPTX(self.nv_prog, self.ptx_buf))

        # Load PTX as module data
        self.ptx = np.char.array(self.ptx_buf)
        self.nv_module = checkCudaErrorsAndReturn(cuda.cuModuleLoadData(self.ptx.ctypes.data))

        module_list.append(self)

    def __getattr__(self, name: str) -> CudaFunction:
        return self.get_function(name)

    def get_function(self, name: str) -> CudaFunction:
        if name in self.fn_cache:
            return self.fn_cache[name]

        fn = CudaFunction(self, name)
        self.fn_cache[name] = fn
        return fn

    def has_function(self, name: str) -> bool:
        try:
            self.get_function(name)
            return True
        except CudaFunctionNameNotFound:
            return False

    @staticmethod
    def find_function(name: str) -> CudaFunction | None:
        for mod in module_list:
            if mod.has_function(name):
                return mod.get_function(name)

        return None

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

    # def __del__(self) -> None:
    #     # if compile fails, the nv_module attribute hasn't been set het
    #     if hasattr(self, "nv_module"):
    #         checkCudaErrors(cuda.cuModuleUnload(self.nv_module))

    @staticmethod
    def from_file(filename: str) -> CudaModule:
        with open(filename) as f:
            code = f.read()
        return CudaModule(code=code, progname=filename)

    @staticmethod
    def clear_list() -> None:
        module_list.clear()


class CudaFunctionCallGraph:
    def __init__(self, g: CudaGraph, fn: CudaFunction, args: Sequence[Any]) -> None:
        self.graph = g
        self.fn = fn

        arg_list = CudaArgList(args, fn.arg_types)

        # create input nodes
        start_nodes = arg_list.create_copy_to_device_nodes(g)

        # TODO: create output nodes

        # create kernel node
        kn = CudaKernelNode(g, fn, arg_list, dependencies=start_nodes)

        # create dependencies
        # print("start_nodes", start_nodes)
        # for n in start_nodes:
        #     kn.depends_on(n)


module_list: list[CudaModule] = []
BlockSpecCallback = Callable[[str, CudaModule, tuple[Any, ...]], BlockSpec]
GridSpecCallback = Callable[[str, CudaModule, tuple[Any, ...]], BlockSpec]
