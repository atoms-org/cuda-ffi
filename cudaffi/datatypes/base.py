from __future__ import annotations

import ctypes
from abc import ABC, abstractmethod
from typing import Any

data_type_registry: DataTypeRegistry = {}
AnyCType = type[ctypes.c_uint] | type[ctypes.c_void_p]


class CudaDataType(ABC):
    def __init__(self, name: str) -> None:
        self.name = str

    @abstractmethod
    def convert(self, data: Any, name: str) -> tuple[int, AnyCType] | None: ...

    @staticmethod
    def register(name: str, DataType: type[CudaDataType], force: bool = False) -> None:
        global data_type_registry
        if name in data_type_registry and not force:
            raise Exception(f"'{name}' already exists as a registered CudaDataType")

        data_type_registry[name] = DataType(name)

    @staticmethod
    def get_registry() -> DataTypeRegistry:
        global data_type_registry
        return data_type_registry


DataTypeRegistry = dict[str, CudaDataType]
