import ctypes

from cudaffi.args import CudaArg, CudaArgDirection


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
        assert not arg.is_pointer
        assert arg.byte_size == 4
        assert arg.direction == CudaArgDirection.input
        assert arg.specified_type is None
        assert arg.ctype == ctypes.c_int
