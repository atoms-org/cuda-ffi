from __future__ import annotations

from collections.abc import Buffer
from enum import Enum, auto
from types import GeneratorType
from typing import Any, Generator

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
    PointerOrHostMem,
)
from .utils import checkCudaErrors


class CudaArgDirection(Enum):
    input = auto()
    output = auto()
    inout = auto()
    autoout = auto()


class CudaArgTypeException(Exception):
    pass


class CudaArgException(Exception):
    pass


class CudaArgType:
    def __init__(
        self,
        name: str = "<<unknown>>",
        type: str | None = None,
        direction: str = "inout",
        byte_size: int | None = None,
    ) -> None:
        try:
            self.direction = CudaArgDirection[direction.lower()]
        except:
            raise CudaArgTypeException(f"Invalid arg direction: '{direction}'")

        if type is None:
            raise CudaArgTypeException("Unspecified arg type")
        self.type = type
        self.name = name
        self.byte_size = byte_size
        self.is_output = (
            self.direction == CudaArgDirection.output
            or self.direction == CudaArgDirection.autoout
            or self.direction == CudaArgDirection.inout
        )
        self.is_input = (
            self.direction == CudaArgDirection.input or self.direction == CudaArgDirection.inout
        )
        self.is_autoout = self.direction == CudaArgDirection.autoout

    def __str__(self) -> str:
        return f"CudaArgType(type='{self.type}',direction='{self.direction}')"

    @staticmethod
    def from_tuple(arg: CudaSimpleArg, name: str = "<<unnamed>>") -> CudaArgType:
        if len(arg) < 2 or len(arg) > 3:
            raise CudaArgTypeException(
                f"Wrong number of args in type specification: expected 2 or 3 got '{arg}'"
            )

        dir = arg[0]
        if not hasattr(CudaArgDirection, dir):
            raise CudaArgTypeException(f"Unknown direction: {dir}")

        t = arg[1]

        byte_size = None
        if dir == "autoout":
            if len(arg) != 3:
                raise CudaArgTypeException(
                    f"'autoout' argument type requires three arguments, got '{arg}'"
                )
            byte_size = arg[2]

        return CudaArgType(name, t, dir, byte_size)


CudaSimpleArg = tuple[str, str] | tuple[str, str, int]
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
        print(f"creating arg for '{data}' with {arg_type}")
        init()

        self.specified_type = arg_type
        self.data = data
        self.nv_data: int | NvDeviceMemory | NvManagedMemory

        # find the datatype
        final_type_str: str | None = None
        data_type_registry = CudaDataType.get_registry()

        # we have a desired arg datatype
        if arg_type is not None:
            final_type_str = arg_type.type
            if final_type_str not in data_type_registry:
                raise Exception(f"'{final_type_str}' is not a registered data type")  # TODO
            datatype = data_type_registry[final_type_str]

            if arg_type.is_input and not datatype.is_type(data):
                raise CudaDataConversionError(
                    data, arg_type, f"data could not be converted to '{final_type_str}'"
                )

            if arg_type.is_autoout:
                print("arg is autoout, should I allocate memory or something?")

        # no desired type, try all of them
        for type in data_type_registry:
            if data_type_registry[type].is_type(data):
                final_type_str = type
                break

        if final_type_str is None and arg_type is not None:
            raise CudaDataConversionError(data, arg_type, f"converter not found for data")

        assert final_type_str is not None
        self.data_type = data_type_registry[final_type_str]
        default_direction = CudaArgDirection[self.data_type.default_direction]
        self.direction: CudaArgDirection = (
            arg_type.direction if arg_type is not None else default_direction
        )
        self.is_output = (
            self.direction == CudaArgDirection.output
            or self.direction == CudaArgDirection.autoout
            or self.direction == CudaArgDirection.inout
        )
        self.is_input = (
            self.direction == CudaArgDirection.input or self.direction == CudaArgDirection.inout
        )
        self.is_autoout = self.direction == CudaArgDirection.autoout
        self.ctype = self.data_type.get_ctype(data)
        self.is_pointer = self.ctype.__name__.endswith("_p")
        self.dev_mem: CudaDeviceMemory | None = None
        if arg_type is not None and arg_type.is_autoout:
            self.byte_size = arg_type.byte_size
            assert self.byte_size is not None
        else:
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

        nv_data: PointerOrHostMem | int
        if isinstance(dec_data, GeneratorType):
            self.data_generator: Generator[PointerOrHostMem, CudaDeviceMemory, Any] | None = (
                dec_data
            )
            host_buf: PointerOrHostMem = next(self.data_generator)
        else:
            self.data_generator = None
            host_buf = dec_data  # type: ignore
            self.output_data = host_buf

        host_nv_data = to_host_nv_data(host_buf)

        checkCudaErrors(
            cuda.cuMemcpyDtoHAsync(
                host_nv_data,
                self.dev_mem.nv_device_memory,
                self.byte_size,
                stream.nv_stream,
            )
        )

    def get_output(self) -> Any:
        if self.direction != CudaArgDirection.autoout:
            raise CudaArgException("Attempting to get output for non-'autoout' argument")

        if self.data_generator is not None:
            try:
                next(self.data_generator)
            except StopIteration as e:
                self.output_data = e.value

        return self.output_data


class CudaArgList:
    def __init__(self, args: tuple[Any], arg_types: CudaArgTypeList | None = None) -> None:
        if arg_types is not None:
            expected_arg_count = len(arg_types)

            # don't expect to pass in autoout args, we will handle those automatically
            for arg in arg_types:
                if arg.direction == CudaArgDirection.autoout:
                    expected_arg_count -= 1

            # ensure we got the right number of inputs
            if len(args) != expected_arg_count:
                raise CudaArgException("Wrong number of arguments")

        # modify args to add autoout args
        if arg_types is not None and len(arg_types) > len(args):
            new_args: list[Any] = []
            curr_arg = 0
            for at in arg_types:
                if at.direction != CudaArgDirection.autoout:
                    new_args.append(args[curr_arg])
                    curr_arg += 1
                else:
                    new_args.append(None)

            args = tuple(new_args)

        # create a CudaArg for every argument
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

        nv_data_args: list[Any] = []
        nv_type_args: list[Any] = []
        print("to_nv_args", len(self.args))
        for arg in self.args:
            nv_data_args.append(arg.nv_data)
            nv_type_args.append(arg.ctype)

        nv_args = (tuple(nv_data_args), tuple(nv_type_args))
        return nv_args

    def get_outputs(self) -> Any:
        ret: list[Any] = []
        for arg in self.args:
            if arg.direction == CudaArgDirection.autoout:
                ret.append(arg.get_output())

        if len(ret) == 0:
            return None
        elif len(ret) == 1:
            return ret[0]
        else:
            return ret


def to_host_nv_data(data: PointerOrHostMem | int) -> int | NvHostMemory | Buffer:
    if isinstance(data, CudaHostMemory):
        return data.nv_host_memory
    elif isinstance(data, int):
        return data
    else:
        return data[0]
