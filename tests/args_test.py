import ctypes

import pytest

from cudaffi.args import CudaArg, CudaArgDirection, CudaArgType, CudaDataTypeWarning


class TestArgs:
    def test_args_str(self) -> None:
        str_arg = "foo"
        arg = CudaArg(str_arg)
        assert arg.is_pointer
        assert arg.dev_mem is not None
        assert arg.data is str_arg
        assert isinstance(arg.nv_data, int)
        assert arg.byte_size == 4
        assert arg.direction == CudaArgDirection.input
        assert arg.specified_type is None
        assert arg.ctype == ctypes.c_void_p

    def test_int(self) -> None:
        arg = CudaArg(42)
        assert not arg.is_pointer
        assert arg.dev_mem is None
        assert arg.data == 42
        assert arg.nv_data == 42
        assert arg.byte_size == 4
        assert arg.direction == CudaArgDirection.input
        assert arg.specified_type is None
        assert arg.ctype == ctypes.c_int

    def test_int_with_type(self) -> None:
        arg = CudaArg(42, CudaArgType(direction="input", type="uint"))
        assert not arg.is_pointer
        assert arg.dev_mem is None
        assert arg.data == 42
        assert arg.nv_data == 42
        assert arg.byte_size == 4
        assert arg.direction == CudaArgDirection.input
        assert arg.specified_type is not None
        assert arg.ctype == ctypes.c_int

    def test_int_sizes(self) -> None:
        a = CudaArg(0, CudaArgType(direction="input", type="int"))
        assert a.byte_size == 4
        a = CudaArg(0, CudaArgType(direction="input", type="uint"))
        assert a.byte_size == 4
        a = CudaArg(0, CudaArgType(direction="input", type="int8"))
        assert a.byte_size == 1
        a = CudaArg(0, CudaArgType(direction="input", type="uint8"))
        assert a.byte_size == 1
        a = CudaArg(0, CudaArgType(direction="input", type="int16"))
        assert a.byte_size == 2
        a = CudaArg(0, CudaArgType(direction="input", type="uint16"))
        assert a.byte_size == 2
        a = CudaArg(0, CudaArgType(direction="input", type="int32"))
        assert a.byte_size == 4
        a = CudaArg(0, CudaArgType(direction="input", type="uint32"))
        assert a.byte_size == 4
        a = CudaArg(0, CudaArgType(direction="input", type="int64"))
        assert a.byte_size == 8
        a = CudaArg(0, CudaArgType(direction="input", type="uint64"))
        assert a.byte_size == 8

    def test_int_small_size_warning(self) -> None:
        with pytest.warns(
            CudaDataTypeWarning, match="passed int value '-200' too small for 1 bytes of int8"
        ):
            CudaArg(-200, CudaArgType(direction="input", type="int8"))

    def test_int_large_size_warning(self) -> None:
        with pytest.warns(
            CudaDataTypeWarning, match="passed int value '300' too large for 1 bytes of int8"
        ):
            CudaArg(300, CudaArgType(direction="input", type="int8"))

    def test_bytearray(self) -> None:
        arg = CudaArg(bytearray([1, 2, 3]))
        assert arg.is_pointer
        assert arg.dev_mem is not None
        assert isinstance(arg.data, bytearray)
        assert arg.data == bytearray([1, 2, 3])
        assert isinstance(arg.nv_data, int)
        assert arg.byte_size == 3
        assert arg.direction == CudaArgDirection.inout
        assert arg.specified_type is None
        assert arg.ctype == ctypes.c_void_p

    def test_bytes(self) -> None:
        arg = CudaArg(bytes([4, 5, 6, 7]))
        assert arg.is_pointer
        assert arg.dev_mem is not None
        assert isinstance(arg.data, bytes)
        assert arg.data == bytes([4, 5, 6, 7])
        assert isinstance(arg.nv_data, int)
        assert arg.byte_size == 4
        assert arg.direction == CudaArgDirection.inout
        assert arg.specified_type is None
        assert arg.ctype == ctypes.c_void_p
