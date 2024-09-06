from typing import Any

from cuda import cuda, nvrtc


def _cudaGetErrorEnum(error: Any) -> Any:
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError("Unknown error type: {}".format(error))


def checkCudaErrors(result: tuple[Any, ...]) -> Any:
    if result[0].value:
        raise RuntimeError(
            "CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0]))
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]
