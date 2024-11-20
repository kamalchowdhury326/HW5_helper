#!/bin/bash
source /opt/asn/etc/asn-bash-profiles-special/modules.sh
module load cuda/11.7.0
nsys profile --stats=true ./transpose