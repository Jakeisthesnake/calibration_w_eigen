#!/bin/bash

# Add Pangolin library to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/jake/calibration_w_eigan/third_party/Pangolin/install/lib:$LD_LIBRARY_PATH

# Manually parse the input string into executable and arguments
set -- $1

# Execute the binary with arguments
exec "$@"
