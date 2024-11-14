#include <iostream>
#include <math.h>

// const int TILE_DIM = 32;
// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

__global__ void add1(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}
__global__ void add2(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}


inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}
int main(void)
{
  int N = 1<<20;  //1M
  float *x, *y;
  
  // Set execution configuration parameters
	//		thr_per_blk: number of CUDA threads per grid block
	//		blk_in_grid: number of blocks in grid
	int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;  
                    //(N-1)/blockSize+1=(((2^20)-1)/2^8)+1=2^12+1=4096
  
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  cudaEvent_t startEvent, stopEvent;
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  float ms;
  checkCuda( cudaEventRecord(startEvent, 0) );

  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  // cudaDeviceSynchronize();
  
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  fprintf(stderr,"\n----------------------------------\n");
  fprintf(stderr,"add<<<1,1>>>  GPU time is taken=%f ms\n",ms);
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
  
  checkCuda( cudaEventRecord(startEvent, 0) );

  add1<<<1, blockSize>>>(N, x, y);    //blockSize=256
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  fprintf(stderr,"\n----------------------------------\n");
  fprintf(stderr,"add1<<<1,256>>>  GPU time is taken=%f ms\n",ms);
  // Check for errors (all values should be 3.0f)
  maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
  
  checkCuda( cudaEventRecord(startEvent, 0) );

  add2<<<numBlocks, blockSize>>>(N, x, y);    //blockSize=256 , numBlocks=4096
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  fprintf(stderr,"\n----------------------------------\n");
  fprintf(stderr,"add2<<<4096,256>>>  GPU time is taken=%f ms\n",ms);
  // Check for errors (all values should be 3.0f)
  maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
  
  
  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}