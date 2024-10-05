from typing import Any

import pytest

from cudaffi.module import CudaModule


@pytest.fixture(scope="function", autouse=True)
def cleanup_after_test() -> Any:
    yield
    CudaModule.clear_list()
