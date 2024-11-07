#!/bin/bash


source /apps/profiles/modules_asax.sh.dyn

module load cuda/11.7.0

# nvcc add_cuda.cu -o add_cuda
# ./add_cuda


nvcc transpose.cu -o transpose
./transpose

