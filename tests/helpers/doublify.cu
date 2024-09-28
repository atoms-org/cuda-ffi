__global__ void doublify(float *a) {
  int idx = threadIdx.x + threadIdx.y * 4;
  printf("idx %d\n", idx);
  printf("idx in: %f\n", a[idx]);
  a[idx] *= 2;
  printf("idx out: %f\n", a[idx]);
}