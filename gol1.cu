#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

#include "cuda_error.h"
#include "default_values.h"
#include "common.h"
#include "cuda_common.h"

//Not pretty, but cc 1.1 doesn't allow to compile kernels in different files, and cc 2.0 does not allow to force noinline. 
#include "cuda_common.cu"


//Los desplazamientos son por los vecinos ghost
/*int iy = blockDim.y * blockIdx.y + threadIdx.y + 1;
int ix = blockDim.x * blockIdx.x + threadIdx.x + 1;
int id = iy * (dim+2) + ix;*/

//Estos define estan porque cuando se usa el data type double se utilizan mas registros y no se puede
//ejecutar la simulacion con bs 1024.
#define my_id (row * (size_i+halo) + col)
#define my_sh_id (threadIdx.x * blockDim.y + threadIdx.y)

__global__ void moveKernel(plattice1D d_lattice, plattice1D d_lattice_new, int size_i, int size_j, int neighs, int halo){
	int count=0;
	int col = (blockDim.x - halo) * blockIdx.x + threadIdx.x;
	int row = (blockDim.y - halo) * blockIdx.y + threadIdx.y; 	
		
	int sh_size_x = blockDim.y;
	
    extern __shared__ TYPE sh_lattice[];

 	if (col < size_i+halo && row < size_j+halo) {
        sh_lattice[my_sh_id] = d_lattice[my_id];
 	}
    __syncthreads();

    // CHECK IF
	/*if (col < size_i+neighs && row < size_j+neighs && 
		threadIdx.x >= (neighs-1) && threadIdx.x < blockDim.x-neighs && 
		threadIdx.y >= (neighs-1) && threadIdx.y < blockDim.y-neighs) {*/
    
    if (col < size_i+neighs && row < size_j+neighs && 
		threadIdx.x >= neighs && threadIdx.x < blockDim.x-neighs && 
		threadIdx.y >= neighs && threadIdx.y < blockDim.y-neighs) {    
        
        count = count_neighs(my_sh_id, sh_size_x-halo, sh_lattice, neighs, halo);	// decrease sh_size_x by 2 to use the same count_neighs function than the rest of the implementations
        check_rules(my_id, count, d_lattice, d_lattice_new);
 	}
}

int main(int argc, char **argv){
	int iter=0;

	timing recorded_time = {.fill = -1, .step_init=-1, .step_end = 0, .comm_init = -1, .comm_end = 0, 
							.evolve = -1, .output_init = -1, .output_end = 0, .total = 0, };
    record_time(10, &recorded_time);

	parameters vars;
	read_input_parameters(argc, argv, &vars);

	size_t size = vars.sizesqr * sizeof(TYPE);

	plattice1D lattice = (plattice1D) malloc(size);
	
	RECORD_TIME(0, &recorded_time);
	fill_lattice(lattice, &vars);
	RECORD_TIME(0, &recorded_time);

	plattice1D d_lattice_new;
	plattice1D d_lattice_tmp;
	plattice1D d_lattice;
	CudaSafeCall(cudaMalloc(&d_lattice, size));
	CudaSafeCall(cudaMalloc(&d_lattice_new, size));
	CudaSafeCall(cudaMemcpy(d_lattice, lattice, size, cudaMemcpyHostToDevice));


	dim3 threadsPerBlock = select_threadsPerBlock(vars.blocksize);
	dim3 numBlocks(ceil((float)vars.size_i / (float)(threadsPerBlock.x-vars.halo)), ceil((float)vars.size_j / (float)(threadsPerBlock.y-vars.halo)));
//	dim3 numBlocks((ceil((float)vars.size_i) / (float)threadsPerBlock.x), ceil((float)vars.size_j / (float)threadsPerBlock.y));

	//dim3 threadsPerBlock(BLOCKSIZE_x, BLOCKSIZE_y, 1);
	//dim3 numBlocks(ceil(vars.size_i / (float)(BLOCKSIZE_x-2)), ceil(vars.size_j / (float)(BLOCKSIZE_y-2)));

	size_t sharedsize = sizeof(TYPE) * (threadsPerBlock.x * threadsPerBlock.y);

	printf("sharedsize=%d\n", sharedsize );

    dim3 ghostBlockSize(vars.blocksize);
    dim3 ghostRowsGridSize((int)ceil(vars.size_i/(float)ghostBlockSize.x));
    dim3 ghostColsGridSize((int)ceil((vars.size_i+vars.halo)/(float)ghostBlockSize.x));


	RECORD_TIME(2, &recorded_time);
	while (iter <= vars.max_iter){
		if (iter % vars.output_steps == 0) {
			RECORD_TIME(4, &recorded_time);
			write_output(iter, lattice, d_lattice, &vars);
			RECORD_TIME(5, &recorded_time);
		}


		RECORD_TIME(6,&recorded_time);
		copy_Rows<<<ghostRowsGridSize, ghostBlockSize>>>(vars.size_i, d_lattice, vars.neighs, vars.halo);
    	copy_Cols<<<ghostColsGridSize, ghostBlockSize>>>(vars.size_i, d_lattice, vars.neighs, vars.halo);
    	RECORD_TIME(7,&recorded_time);
    	
    	RECORD_TIME(8,&recorded_time);
		moveKernel<<<numBlocks, threadsPerBlock, sharedsize>>>(d_lattice, d_lattice_new, vars.size_i, vars.size_j, vars.neighs, vars.halo);
		cudaDeviceSynchronize();
		CudaCheckError();

		d_lattice_tmp = &d_lattice[0];
		d_lattice = &d_lattice_new[0];
		d_lattice_new = &d_lattice_tmp[0];
		
		RECORD_TIME(9,&recorded_time);

		iter++;
	}
	RECORD_TIME(2, &recorded_time);

	cudaDeviceReset();
	free(lattice);

	record_time(10, &recorded_time);
	
	output_information(&vars, &recorded_time);
	return 0;
}