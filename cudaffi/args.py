from __future__ import annotations

from collections.abc import Buffer
from enum import Enum, auto
from types import GeneratorType
from typing import Any

from cuda import cuda

from .core import CudaStream, init
from .memory import (
    AnyCType,
    CudaDataType,
    CudaDeviceMemory,
    CudaHostMemory,
    NvDeviceMemory,
    NvHostMemory,
    NvManagedMemory,
    PointerGenerator,
    PointerOrHostMem,
)
from .utils import checkCudaErrors


class CudaArgDirection(Enum):
    input = auto()
    output = auto()
    inout = auto()
    autoout = auto()


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
        init()

        self.specified_type = arg_type
        self.data = data
        self.nv_data: int | NvDeviceMemory | NvManagedMemory

        # find the datatype
        final_type_str: str | None = None
        data_type_registry = CudaDataType.get_registry()

        # we have a desired arg datatype
        if arg_type is not None:
            arg_type_str = arg_type.type
            if arg_type_str not in data_type_registry:
                raise Exception(f"'{arg_type_str}' is not a registered data type")  # TODO
            datatype = data_type_registry[arg_type_str]

            if datatype.is_type(data):
                final_type_str = arg_type.type

        # no desired type, try all of them
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

        self.data_type = data_type_registry[final_type_str]
        default_direction = CudaArgDirection[self.data_type.default_direction]
        self.direction: CudaArgDirection = (
            arg_type.direction if arg_type is not None else default_direction
        )
        self.ctype = self.data_type.get_ctype(data)
        self.is_pointer = self.ctype.__name__.endswith("_p")
        self.dev_mem: CudaDeviceMemory | None = None
        self.byte_size = self.data_type.get_byte_size(data)

        if self.is_pointer:
            self.dev_mem = CudaDeviceMemory(self.byte_size)
            self.nv_data = self.dev_mem.nv_device_memory
        else:
            self.nv_data = data

    def copy_to_device(self, stream: CudaStream | None = None) -> None:
        if not self.is_pointer:
            return

        if stream is None:
            stream = CudaStream.get_default()

        assert self.dev_mem is not None

        enc_data = self.data_type.encode(self.data)
        host_nv_data = to_host_nv_data(enc_data)

        checkCudaErrors(
            cuda.cuMemcpyHtoDAsync(
                self.dev_mem.nv_device_memory,
                host_nv_data,
                self.byte_size,
                stream.nv_stream,
            )
        )

    def copy_to_host(self, stream: CudaStream | None = None) -> None:
        if not self.is_pointer:
            return

        if stream is None:
            stream = CudaStream.get_default()

        assert self.dev_mem is not None

        dec_data = self.data_type.decode(self.data)
        # reveal_type(dec_data)

        nv_data: PointerOrHostMem | int
        if isinstance(dec_data, GeneratorType):
            print("generator")
            gen: PointerGenerator[Any] = dec_data
            # prime the generator
            next(gen)
            # get the only value from the generator
            try:
                ret = next(gen)
            except StopIteration as e:
                ret = e.value
        else:
            print("non-generator")
            ret = dec_data  # type: ignore
        print("ret type", type(ret))
        host_nv_data = to_host_nv_data(ret)

        checkCudaErrors(
            cuda.cuMemcpyDtoHAsync(
                host_nv_data,
                self.dev_mem.nv_device_memory,
                self.byte_size,
                stream.nv_stream,
            )
        )


class CudaArgList:
    def __init__(self, args: tuple[Any], arg_types: CudaArgTypeList | None = None) -> None:
        if arg_types is not None and len(args) != len(arg_types):
            raise Exception("Wrong number of arguments")  # TODO

        self.args: list[CudaArg] = []
        for n in range(len(args)):
            arg = args[n]
            arg_type = arg_types[n] if arg_types is not None else None
            self.args.append(CudaArg(arg, arg_type))

    def copy_to_device(self) -> None:
        for arg in self.args:
            if arg.direction == CudaArgDirection.input or arg.direction == CudaArgDirection.inout:
                arg.copy_to_device()

    def copy_to_host(self) -> None:
        for arg in self.args:
            if (
                arg.direction == CudaArgDirection.output
                or arg.direction == CudaArgDirection.autoout
                or arg.direction == CudaArgDirection.inout
            ):
                arg.copy_to_host()

    def to_nv_args(self) -> NvKernelArgs:
        if len(self.args) == 0:
            return 0

        # nv_data_args = tuple([arg.nv_data for arg in self.args])
        # nv_type_args = tuple([arg.ctype for arg in self.args])
        nv_data_args: list[Any] = []
        nv_type_args: list[Any] = []
        for arg in self.args:
            nv_data_args.append(arg.nv_data)
            print("adding arg", arg.nv_data)
            nv_type_args.append(arg.ctype)

        print("nv_data_args", nv_data_args)
        nv_args = (tuple(nv_data_args), tuple(nv_type_args))
        # nv_args = (nv_data_args, nv_type_args)
        return nv_args


def to_host_nv_data(data: PointerOrHostMem | int) -> int | NvHostMemory | Buffer:
    if isinstance(data, CudaHostMemory):
        return data.nv_host_memory
    elif isinstance(data, int):
        return data
    else:
        return data[0]
