from ..memory import CudaDataType

__all__ = ["CudaDataType"]


def init() -> None:
    from .str_type import CudaStrDataType

    CudaDataType.register("str", CudaStrDataType, force=True)
