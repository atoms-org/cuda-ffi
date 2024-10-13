import ctypes
from collections.abc import Buffer
from typing import (
    TYPE_CHECKING,
    Any,
    NewType,
    Sequence,
    TypeAlias,
    TypeVar,
    TypeVarTuple,
    cast,
    overload,
)

from cuda import cuda, cudart, nvrtc

# TODO: remove from module.py and args.py
BlockSpec = tuple[int, int, int]
GridSpec = tuple[int, int, int]
if TYPE_CHECKING:
    AnyCType: TypeAlias = type[ctypes._SimpleCData[Any]]
else:
    AnyCType: TypeAlias = ctypes._SimpleCData
NvDevPtr = NewType("NvDevPtr", int)
ArgsSpec = int | tuple[Sequence[Buffer | NvDevPtr], Sequence[AnyCType]]
NvErrorType = cuda.CUresult | cudart.cudaError_t | nvrtc.nvrtcResult

T = TypeVar("T")
T2 = TypeVar("T2")
Ts = TypeVarTuple("Ts")


def _cudaGetErrorEnum(error: int) -> Any:
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError("Unknown error type: {}".format(error))


def checkCudaErrorsNoReturn(result: tuple[NvErrorType]) -> None:
    assert len(result) == 1
    if result[0].value:
        raise RuntimeError(
            "CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0]))
        )


@overload
def checkCudaErrorsAndReturn(result: tuple[NvErrorType, T]) -> T: ...


@overload
def checkCudaErrorsAndReturn(result: tuple[NvErrorType, T, T2, *Ts]) -> tuple[T, T2, *Ts]: ...


def checkCudaErrorsAndReturn(
    result: tuple[NvErrorType, T] | tuple[NvErrorType, *Ts],
) -> T | tuple[*Ts]:
    assert len(result) >= 2
    if result[0].value:
        raise RuntimeError(
            "CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0]))
        )

    if len(result) == 2:
        assert len(result) > 1
        return cast(T, result[1])
    else:
        return result[1:]
