__global__ void print_buf(char *buf, int len) {
  int i;

  printf("buf is %p\n", buf);
  printf("len is %d\n", len);
  printf("buf[0] is %d\n", buf[0]);

  for (i = 0; i < len; i++) {
    printf("buf: %d\n", buf[i]);
  }
  printf("buf as string: %s\n", buf);
  printf("done.\n");
}