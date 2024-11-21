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


__global__ void moveKernel(plattice1D d_lattice, plattice1D d_lattice_new, int size_i, int size_j, int neighs, int halo){
	int count=0;
	//Los desplazamientos son por los vecinos ghost
	int col = blockDim.x * blockIdx.x + threadIdx.x + neighs;
	int row = blockDim.y * blockIdx.y + threadIdx.y + neighs;
	int my_id = row * (size_i+halo) + col; 

	if (col < size_i+neighs && row < size_j+neighs){
		count = count_neighs(my_id, size_i, d_lattice, neighs, halo);
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
 	
 	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	
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
	//dim3 numBlocks(((vars.size_i) / threadsPerBlock.x), ((vars.size_j) / threadsPerBlock.y));
 	dim3 numBlocks(ceil(((float)vars.size_i) / (float)threadsPerBlock.x), ceil((float)vars.size_j / (float)threadsPerBlock.y));

    dim3 ghostBlockSize(vars.blocksize);
    dim3 ghostRowsGridSize((int)ceil(vars.size_i/(float)ghostBlockSize.x));
    dim3 ghostColsGridSize((int)ceil((vars.size_j+vars.halo)/(float)ghostBlockSize.x));


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

		//write_output(iter, lattice, d_lattice, &vars);


    	RECORD_TIME(8,&recorded_time);
		moveKernel<<<numBlocks, threadsPerBlock>>>(d_lattice, d_lattice_new, vars.size_i, vars.size_j, vars.neighs, vars.halo);
		cudaThreadSynchronize();
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