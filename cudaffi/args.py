from __future__ import annotations

import ctypes
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAlias

from .core import init
from .memory import CudaDataType, CudaDeviceMemory, CudaMemory, NvDeviceMemory, NvManagedMemory


class CudaArgDirection(Enum):
    input = 1
    output = 2
    inout = 3


CudaSimpleArg = tuple[str, str]


class CudaArgType:
    def __init__(
        self,
        name: str = "<<unknown>>",
        type: str | None = None,
        direction: str = "inout",
    ) -> None:
        try:
            self.direction = CudaArgDirection[direction.lower()]
        except:
            raise Exception(f"Invalid arg direction: '{direction}'")  # TODO

        if type is None:
            raise Exception("Unspecified arg type")  # TODO
        self.type = type
        self.name = name

    @staticmethod
    def from_tuple(arg: CudaSimpleArg, name: str = "<<unknown>>") -> CudaArgType:
        dir = arg[0]
        t = arg[1]
        return CudaArgType(name, t, dir)


CudaArgTypeList = list[CudaArgType]
CudaArgSpec = dict[str, Any] | CudaSimpleArg
CudaArgSpecList = list[CudaArgSpec]

# XXX - https://mypy.readthedocs.io/en/stable/runtime_troubles.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
if TYPE_CHECKING:
    AnyCType: TypeAlias = type[ctypes._SimpleCData[Any]]
else:
    AnyCType: TypeAlias = ctypes._SimpleCData

NvKernelArgs = (
    int  # for None args
    | tuple[
        tuple[Any, ...],  # list of data
        tuple[AnyCType, ...],  # list of data types
    ]
)


class CudaDataConversionError(Exception):
    def __init__(self, data: Any, arg_type: CudaArgType | None, msg: str):
        super().__init__(msg)
        self.data = data
        self.arg_type = arg_type


class CudaArg:
    def __init__(
        self,
        data: Any,
        arg_type: CudaArgType | None = None,
        # data_type: CudaDataType[Any] | None = None,
    ) -> None:
        self.arg_type = arg_type
        self.data = data
        self.nv_data: int | NvDeviceMemory | NvManagedMemory
        self.ctype: AnyCType
        self.mem: CudaMemory | None

        init()

        # find the arg datatype
        final_type_str: str | None = None
        data_type_registry = CudaDataType.get_registry()
        if arg_type is not None:
            arg_type_str = arg_type.type
            if arg_type_str not in data_type_registry:
                raise Exception(f"'{arg_type_str}' is not a registered data type")  # TODO
            datatype = data_type_registry[arg_type_str]

            if datatype.is_type(data):
                final_type_str = arg_type.type

        for type in data_type_registry:
            if data_type_registry[type].is_type(data):
                final_type_str = type
                break

        if final_type_str is None:
            if arg_type is not None:
                raise CudaDataConversionError(
                    data, arg_type, f"data could not be converted to '{arg_type.type}'"
                )
            else:
                raise CudaDataConversionError(data, arg_type, f"converter not found for data")

        final_type = data_type_registry[final_type_str]
        # arg type was specified by argument types
        if arg_type is not None:
            mem = CudaDeviceMemory(final_type.get_byte_size(data))
            self.nv_data = mem.dev_addr
            self.ctype = mem.ctype
        # arg is CudaMemory type
        elif isinstance(data, CudaMemory):
            self.nv_data = data.dev_addr
            self.ctype = data.ctype
        # arg is ctype data
        elif isinstance(data, ctypes._SimpleCData):
            self.nv_data = data.value
            self.ctype = data.__class__
        # arg is raw int
        elif isinstance(data, int):
            self.nv_data = data
            self.ctype = ctypes.c_int64
        # don't know what arg is, try to convert it
        else:
            mem = CudaDeviceMemory(final_type.get_byte_size(data))
            self.nv_data = mem.dev_addr
            self.ctype = mem.ctype

    def to_device(self) -> None:
        pass

    def to_host(self) -> None:
        pass


class CudaArgList:
    def __init__(self, args: tuple[Any], arg_types: CudaArgTypeList | None = None) -> None:
        if arg_types is not None and len(args) != len(arg_types):
            raise Exception("Wrong number of arguments")  # TODO

        self.args: list[CudaArg] = []
        for n in range(len(args)):
            arg = args[n]
            arg_type = arg_types[n] if arg_types is not None else None
            self.args.append(CudaArg(arg, arg_type))

    def to_device(self) -> None:
        pass

    def to_host(self) -> None:
        pass

    def to_nv_args(self) -> NvKernelArgs:
        if len(self.args) == 0:
            return 0

        nv_data_args = tuple([arg.nv_data for arg in self.args])
        nv_type_args = tuple([arg.ctype for arg in self.args])
        nv_args = (nv_data_args, nv_type_args)
        return nv_args
