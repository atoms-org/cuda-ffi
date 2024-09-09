from .base import CudaDataType
from .int_type import CudaIntDataType
from .memory_type import CudaMemoryDataType

__all__ = ["CudaDataType"]

CudaDataType
CudaDataType.register("int", CudaIntDataType)
CudaDataType.register("pointer", CudaMemoryDataType)
