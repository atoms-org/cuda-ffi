from ..memory import CudaDataType

__all__ = ["CudaDataType"]


def init() -> None:
    # from .array_type import CudaArrayDataType
    # from .bytearray_type import CudaByteArrayDataType
    # from .bytes_type import CudaBytesDataType
    # from .numpy_type import CudaNumpyDataType
    from .str_type import CudaStrDataType

    # CudaDataType.register("array", CudaArrayDataType, force=True)
    # CudaDataType.register("bytes", CudaBytesDataType, force=True)
    # CudaDataType.register("bytearray", CudaByteArrayDataType, force=True)
    # CudaDataType.register("numpy", CudaNumpyDataType, force=True)
    CudaDataType.register("str", CudaStrDataType, force=True)
