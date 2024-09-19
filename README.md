# CUDA-FFI
A NVIDIA CUDA Foreign Function Interface (FFI) library.

## Features
- Run CUDA code in Python from a string or a file
- Follows common Foreign Function Interface (FFI) design pattern like ctypes or node-ffi
- Automatic argument conversion and data transfer
    - Supported argument types: `int`, `array`, `bytes`, `bytearray`, `string`,
      `numpy`, any object implementing the [buffer protocol](https://docs.python.org/3/c-api/buffer.html)
    - Extensible arugument typing through `CudaDataType.register()`
- Argument type checking to prevent errors
- Chaining of CUDA kernels and memory transfers in a graph for highest efficiency


## Simple defaults
``` python
from cudaffi import CudaModule

mod = CudaModule("""
__global__ void tryme(char *str) {
    printf("string is: %s\n", str);
}
""")
mod.tryme("this is a test")
# string is: this is a test
```

## Optional Arg Checking, Conversion
``` python
from cudaffi import CudaModule

mod = CudaModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")
# arg type checking and memory transfer directions
mod.multiply_them.arg_types = [("output", "numpy"), ("input", "numpy"), ("input", "numpy")]

a = numpy.random.randn(400).astype(numpy.float32)
b = numpy.random.randn(400).astype(numpy.float32)
out = numpy.zeros_like(a)

# type checking
mod.multiply_them(out, a, [1,2,3,4])
# TypeError: expected numpy for arg1

# manually set block and grid size
mod.multiply_them(out, a, b, block=(400,1,1), grid=(1,1,1))
```

## Optional Validation, Automatic Block & Grid Size, Return Types
``` python
# use 'autoout' and specify size to automatically allocate and convert return results in the desired format
mod.multiply_them.arg_types = [("output", "numpy": lambda args: arg1[1].size), ("input", "numpy"), ("input", "numpy")]

# validate that the input arrays have the same shape
mod.multiply_them.validate_args = lambda args: args[1].shape == args[2].shape

# default CUDA block size is the size of the input array
mod.multiply_them.default_block = lambda args: (args[1].size, 1, 1)

# return type created automatically from autoout... if multiple autoouts are defined, a tuple of results is returned
out = mod.multiply_them(a, b)
```

## Graphs and Chaining
``` python
from cudaffi import CudaModule, CudaGraph

mod = CudaModule.load_file("test_graph.cu")
mod.start_ker.arg_types = [("input", "numpy"), ("output", "bytes"), ("output", "int32")]
mod.middle_ker.arg_types = [("input", "bytes"), ("input", "int32"), ("output", "bytes"), ("output", "int16")]
mod.end_ker.arg_types = [("input", "bytes"), ("input", "int16"), ("output", "int16")]

CudaGraph.start(mod.start_ker)
```

## NVIDIA Device, Stream, Context Management
``` python
from cudaffi import CudaDevice, CudaStream, CudaContext
CudaDevice.set_default(0)
print("Device:", d.name, d.compute_capability, d.driver_version)

s = CudaStream()
CudaStream.set_default(s)
ctx = CudaContext()
CudaContext.set_default(ctx)
# ...
d = CudaDevice.get_default()
s = CudaStream.get_default()
ctx = CudaContext.get_default()
```

## Adding New DataTypes
See `cudaffi/datatypes/*.py` for examples