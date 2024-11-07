#!/bin/bash


source /apps/profiles/modules_asax.sh.dyn

module load cuda/11.7.0
# module load cuda11.4/blas/11.4.2
# module load cuda11.4/fft/11.4.2
# module load cuda11.4/nsight/11.4.2
# module load cuda11.4/profiler/11.4.2
# module load cuda11.4/toolkit/11.4.2


nvcc hello_world.cu -o hello_world
./hello_world
# nvcc -DDEBUG0 -o run.out $1
# ./run.out  $2 $3
# ./run.out  $2 $3
# ./run.out  $2 $3


