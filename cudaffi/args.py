from __future__ import annotations

from enum import Enum
from typing import Any


class CudaArgDirection(Enum):
    input = 1
    output = 2
    inout = 3


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
