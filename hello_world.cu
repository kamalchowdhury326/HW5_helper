#include <stdio.h>
#include<cuda.h>

__global__ void helloCUDA()
{
    printf("Hello, CUDA!\n");
}

int main()
{
    helloCUDA<<<2, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}