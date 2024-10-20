from __future__ import annotations

import ctypes
from abc import ABC, abstractmethod
from collections.abc import Buffer
from enum import Enum, auto
from types import GeneratorType
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Generic,
    Iterator,
    Sequence,
    TypeAlias,
    TypeVar,
)

from cuda import cuda, cudart

from .device import CudaStream, init
from .graph.graph import CudaGraph, GraphNode
from .graph.memcpy import CudaMemcpyNode
from .memory import (
    CudaDeviceMemory,
    HostBuffer,
    PointerOrHostMem,
    PointerOrPointerGenerator,
)
from .utils import checkCudaErrorsNoReturn

# XXX - https://mypy.readthedocs.io/en/stable/runtime_troubles.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
if TYPE_CHECKING:
    AnyCType: TypeAlias = type[ctypes._SimpleCData[Any]]
else:
    AnyCType: TypeAlias = ctypes._SimpleCData

data_type_registry: DataTypeRegistry = {}
DataType = TypeVar("DataType")


class CudaDataType(ABC, Generic[DataType]):
    aliases: list[str] = list()
    default_direction: str = "inout"

    def __init__(self, name: str) -> None:
        self.name = str

    @abstractmethod
    def is_type(self, data: Any) -> bool: ...

    @abstractmethod
    def get_byte_size(self, data: DataType) -> int: ...

    @abstractmethod
    def get_ctype(self, data: DataType) -> AnyCType: ...

    @abstractmethod
    def encode(self, data: DataType) -> PointerOrHostMem | int | float | bool: ...

    @abstractmethod
    def decode(
        self, data: DataType | None = None, size_hint: int | None = None
    ) -> PointerOrPointerGenerator[DataType]: ...

    @staticmethod
    def register(
        name: str,
        DataType: type[CudaDataType[Any]],
        force: bool = False,
        is_alias: bool = False,
    ) -> None:
        if name in data_type_registry and not force:
            raise CudaArgTypeException(f"'{name}' already exists as a registered CudaDataType")

        dt = DataType(name)
        data_type_registry[name] = dt
        if hasattr(dt, "aliases") and not is_alias:
            for alias in dt.aliases:
                DataType.register(alias, DataType, is_alias=True, force=force)

    @staticmethod
    def get_registry() -> DataTypeRegistry:
        global data_type_registry
        return data_type_registry

    @staticmethod
    def resolve(
        data: Any,
        arg_type: CudaArgType | None = None,
    ) -> CudaDataType[Any]:
        final_type_str: str | None = None
        data_type_registry = CudaDataType.get_registry()

        # we have a desired arg datatype
        if arg_type is not None:
            final_type_str = arg_type.type
            if final_type_str not in data_type_registry:
                raise CudaArgTypeException(
                    f"'{final_type_str}' is not a registered data type"
                )  # TODO
            datatype = data_type_registry[final_type_str]

            if arg_type.is_input and not datatype.is_type(data):
                raise CudaDataConversionError(
                    data, arg_type, f"data could not be converted to '{final_type_str}'"
                )

        # no desired type, try all of them
        if final_type_str is None:
            for type in data_type_registry:
                if data_type_registry[type].is_type(data):
                    final_type_str = type
                    break

        # finalize the data type
        if final_type_str is None:
            raise CudaDataConversionError(data, arg_type, f"converter not found for data: {data}")

        return data_type_registry[final_type_str]


DataTypeRegistry = dict[str, CudaDataType[Any]]


class CudaArgDirection(Enum):
    input = auto()
    output = auto()
    inout = auto()
    autoout = auto()

    @staticmethod
    def is_input(d: CudaArgDirection) -> bool:
        return (d == CudaArgDirection.input) or (d == d == CudaArgDirection.inout)

    @staticmethod
    def is_output(d: CudaArgDirection) -> bool:
        return (
            (d == CudaArgDirection.output)
            or (d == d == CudaArgDirection.inout)
            or (d == d == CudaArgDirection.autoout)
        )

    @staticmethod
    def is_autoout(d: CudaArgDirection) -> bool:
        return d == CudaArgDirection.autoout


class CudaArgTypeException(Exception):
    pass


class CudaArgException(Exception):
    pass


class CudaDataTypeWarning(RuntimeWarning):
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
        self.is_output = CudaArgDirection.is_output(self.direction)
        self.is_input = CudaArgDirection.is_input(self.direction)
        self.is_autoout = CudaArgDirection.is_autoout(self.direction)

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
        tuple[Buffer | int, ...],  # list of data
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
    ) -> None:
        init()

        self.specified_type = arg_type
        self.data = data
        self.nv_data: int | float | bool | cudart.cudaDevPtr
        self.data_type = CudaDataType.resolve(data, arg_type)
        self.ctype = self.data_type.get_ctype(data)

        # set the arg direction
        default_direction = CudaArgDirection[self.data_type.default_direction]
        self.direction: CudaArgDirection = (
            arg_type.direction if arg_type is not None else default_direction
        )
        self.is_output = CudaArgDirection.is_output(self.direction)
        self.is_input = CudaArgDirection.is_input(self.direction)
        self.is_autoout = CudaArgDirection.is_autoout(self.direction)

        # do memory management
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
            enc_data = self.data_type.encode(self.data)
            assert (
                isinstance(enc_data, int)
                or isinstance(enc_data, float)
                or isinstance(enc_data, bool)
            )
            self.nv_data = enc_data

    def copy_to_device(self, stream: CudaStream | None = None) -> None:
        if not self.is_pointer:
            return

        if stream is None:
            stream = CudaStream.get_default()

        assert self.dev_mem is not None

        enc_data = self.data_type.encode(self.data)
        assert not isinstance(enc_data, int)
        assert not isinstance(enc_data, float)
        assert not isinstance(enc_data, bool)
        buf = HostBuffer(enc_data)

        assert self.byte_size is not None
        checkCudaErrorsNoReturn(
            cuda.cuMemcpyHtoDAsync(
                self.dev_mem.nv_device_memory,
                buf.to_host_nv_data(),
                self.byte_size,
                stream.nv_stream,
            )
        )

    def create_copy_to_device_node(self, g: CudaGraph) -> CudaMemcpyNode | None:
        enc_data = self.data_type.encode(self.data)
        assert not isinstance(enc_data, int | float | bool)
        buf = HostBuffer(enc_data)

        assert self.byte_size is not None
        assert self.dev_mem is not None
        n = CudaMemcpyNode(g, buf, self.dev_mem, self.byte_size)
        return n

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
            host_buf = HostBuffer(next(self.data_generator))
        else:
            self.data_generator = None
            host_buf = HostBuffer(dec_data)  # type: ignore
            self.output_data = host_buf

        assert self.byte_size is not None
        checkCudaErrorsNoReturn(
            cuda.cuMemcpyDtoHAsync(
                host_buf.to_host_nv_data(),
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
    def __init__(self, args: Sequence[Any], arg_types: CudaArgTypeList | None = None) -> None:
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

            args = new_args

        # create a CudaArg for every argument
        self.args: list[CudaArg] = []
        for n in range(len(args)):
            arg = args[n]
            arg_type = arg_types[n] if arg_types is not None else None
            self.args.append(CudaArg(arg, arg_type))

    def __iter__(self) -> Iterator[CudaArg]:
        return iter(self.args)

    def __getitem__(self, idx: int) -> CudaArg:
        return self.args[idx]

    def copy_to_device(self) -> None:
        for arg in self.args:
            if arg.direction == CudaArgDirection.input or arg.direction == CudaArgDirection.inout:
                arg.copy_to_device()

    def create_copy_to_device_nodes(self, g: CudaGraph) -> list[GraphNode]:
        ret: list[GraphNode] = []

        for arg in self.args:
            if arg.is_pointer and (
                arg.direction == CudaArgDirection.input or arg.direction == CudaArgDirection.inout
            ):
                n = arg.create_copy_to_device_node(g)
                if n is not None:
                    ret.append(n)

        return ret

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
