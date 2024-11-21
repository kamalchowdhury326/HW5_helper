#!/bin/bash
source /opt/asn/etc/asn-bash-profiles-special/modules.sh
module load cuda/11.7.0

nvcc stencil1D.cu -o stencil1D
./stencil1D