from typing import Any

import numpy as np
import pytest

from cudaffi.args import CudaDataConversionError
from cudaffi.core import init
from cudaffi.module import (
    BlockSpec,
    CudaCompilationError,
    CudaCompilationWarning,
    CudaFunction,
    CudaFunctionNameNotFound,
    CudaModule,
)


class TestModule:
    def test_exists(self) -> None:
        CudaModule.from_file("tests/helpers/simple.cu")

    def test_compilation_error(self) -> None:
        with pytest.raises(CudaCompilationError, match='error: expected a ";"'):
            CudaModule(
                """
            __global__ void thingy() {
            printf("this is a test\\n") // missing semicolon
            }
            """
            )

    def test_compilation_warning(self) -> None:
        with pytest.warns(
            CudaCompilationWarning, match="the format string requires additional arguments"
        ):
            CudaModule(
                """
            __global__ void thingy() {
            printf("this is a test %d\\n"); // missing argument
            }
            """
            )

    def test_no_extern(self) -> None:
        mod = CudaModule(
            """
            extern "C" __global__ void thingy() {
            printf("this is a test\\n");
            }
            """,
            no_extern=True,
        )
        fn = mod.get_function("thingy")
        assert isinstance(fn, CudaFunction)

    def test_compile_options(self) -> None:
        mod = CudaModule(
            """
            __global__ void thingy() {
            printf("this is a test\\n");
            }
            """,
            compile_options=["--fmad=false"],
        )
        assert len(mod.compile_args) == 2
        assert mod.compile_args[0] == b"--fmad=false"

    def test_include_paths(self) -> None:
        mod = CudaModule(
            """
            #include "dummy.h"

            __global__ void thingy() {
            printf("this is a test: %d\\n", DUMMY);
            }
            """,
            include_dirs=["tests/helpers/include"],
        )
        assert len(mod.compile_args) == 3
        assert mod.compile_args[1] == b"-I"
        assert mod.compile_args[2] == b"tests/helpers/include"

    def test_bad_compile_option(self) -> None:
        with pytest.raises(
            CudaCompilationError, match="unrecognized option --thisisanoptionthatdoesnotexist"
        ):
            CudaModule(
                """
                __global__ void thingy() {
                printf("this is a test\\n");
                }
                """,
                compile_options=["--thisisanoptionthatdoesnotexist"],
            )

    def test_function_cache(self) -> None:
        mod = CudaModule(
            """
        __global__ void thingy() {
          printf("this is a test\\n");
        }
        """
        )
        thing1 = mod.thingy
        thing2 = mod.thingy
        assert thing1 is thing2


class TestFunction:
    def test_basic(self) -> None:
        init(force=True)
        mod = CudaModule(
            """
        __global__ void thingy() {
          printf("this is a test\\n");
        }
        """
        )
        fn = mod.get_function("thingy")
        assert isinstance(fn, CudaFunction)
        mod.thingy()

    def test_from_file(self) -> None:
        init(force=True)
        mod = CudaModule.from_file("tests/helpers/simple.cu")
        fn = mod.get_function("simple")
        assert isinstance(fn, CudaFunction)
        mod.simple()

    def test_one_arg(self) -> None:
        init(force=True)
        mod = CudaModule.from_file("tests/helpers/one_arg.cu")
        fn = mod.get_function("one")
        assert isinstance(fn, CudaFunction)
        mod.one(1)

    def test_wrong_name(self) -> None:
        mod = CudaModule.from_file("tests/helpers/one_arg.cu")
        with pytest.raises(CudaFunctionNameNotFound):
            mod.doesnotexist(1)

    def test_str(self) -> None:
        mod = CudaModule.from_file("tests/helpers/simple.cu")
        s = str(mod.simple)
        assert s == "tests/helpers/simple.cu:simple"

    def test_repr(self) -> None:
        mod = CudaModule.from_file("tests/helpers/simple.cu")
        s = repr(mod.simple)
        assert s.startswith("tests/helpers/simple.cu:simple:0x")

    def test_arg_type_string(self) -> None:
        mod = CudaModule.from_file("tests/helpers/string_arg.cu")
        mod.printstr("blah")

    def test_default_block_tuple(self) -> None:
        mod = CudaModule.from_file("tests/helpers/string_arg.cu")
        mod.printstr.default_block = (1, 1, 1)

    def test_default_block_fn(self) -> None:
        mod = CudaModule.from_file("tests/helpers/string_arg.cu")

        def default_block_fn(name: str, mod: CudaModule, args: Any) -> BlockSpec:
            print("args", args)
            return (1, 1, 1)

        mod.printstr.default_block = default_block_fn

    def test_simple_arg_types(self) -> None:
        mod = CudaModule.from_file("tests/helpers/string_arg.cu")
        mod.printstr.arg_types = [("input", "str")]
        mod.printstr("whee")

    def test_simple_bad_arg(self) -> None:
        mod = CudaModule.from_file("tests/helpers/string_arg.cu")
        mod.printstr.arg_types = [("input", "str")]
        with pytest.raises(CudaDataConversionError, match="could not be converted to 'str'"):
            mod.printstr(21)

    def test_strin_strout(self) -> None:
        mod = CudaModule.from_file("tests/helpers/strstr.cu")
        ba = bytearray(14)
        mod.strstr("input string", ba)
        assert ba.decode() == "this is a test"

    def test_inout(self) -> None:
        mod = CudaModule.from_file("tests/helpers/doublify.cu")
        arr = np.random.randn(4, 4).astype(np.float32)
        print("input", arr)
        mod.doublify(arr, block=(4, 4, 1))
        print("output", arr)

    def test_autoout(self) -> None:
        mod = CudaModule.from_file("tests/helpers/strstr.cu")
        mod.strstr.arg_types = [("input", "str"), ("autoout", "str", 64)]
        retstr = mod.strstr("input string")
        assert retstr == "this is a test"
