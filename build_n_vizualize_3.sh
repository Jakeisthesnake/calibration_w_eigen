#!/bin/bash

# Clean and create build directory
rm -rf build
mkdir build
cd build

# Build the project
cmake ..
make -j$(nproc)

# Set correct library path
export LD_LIBRARY_PATH=/home/jake/calibration_w_eigen/third_party/Pangolin/install/lib:$LD_LIBRARY_PATH

# Run the executable
./vizualize_3_cameras_data