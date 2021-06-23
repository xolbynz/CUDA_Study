#ifndef CAFFE_COMMON_CUH_
#define CAFFE_COMMON_CUH_
#include <stdio.h>
#include "HelloWorld.h"

// __global__ functions, or "kernels", execute on the device
__global__ void hello_kernel()
{
  printf("Hello, world from the device!\n");
}

int main(void)
{
  // greet from the host
  printf("Hello, world from the host!\n");

  // launch a kernel with a single thread to greet from the device
  hello_kernel<<<1,1>>>();

  // wait for the device to finish so that we see the message
  cudaDeviceSynchronize();

  return 0;
}

#endif