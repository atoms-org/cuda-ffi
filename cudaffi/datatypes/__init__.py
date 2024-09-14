from ..memory import CudaDataType

__all__ = ["CudaDataType"]


def init() -> None:
    from .numpy_type import CudaNumpyDataType
    from .str_type import CudaStrDataType

    CudaDataType.register("str", CudaStrDataType, force=True)
    CudaDataType.register("numpy", CudaNumpyDataType, force=True)
