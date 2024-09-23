__device__ int strlen(const char *str) {
  int i = 0;
  while (str[i] != 0)
    i++;
  return i + 1;
}

__global__ void strstr(char *str_in, char *str_out) {
  printf("msg in: %s\n", str_in);
  const char *msg_out = "this is a test";
  printf("len msg_out: %d\n", strlen(msg_out));
  memcpy(str_out, msg_out, 16);
  printf("message is: %s\n", msg_out);
  printf("memcpy done, returning message: %s\n", str_out);
}