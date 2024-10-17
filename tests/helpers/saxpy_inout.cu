__global__ void saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("n: %d\n", n);
  // printf("a: %f\n", a);
  if (i < n) {
    // printf("x[i]: %f\n", x[i]);
    // printf("y[i]: %f\n", y[i]);
    // printf("val: %f\n", a * x[i] + y[i]);
    y[i] = a * x[i] + y[i];
  }
}