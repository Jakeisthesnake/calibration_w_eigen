#!/bin/bash

# Clean and create build directory
rm -rf build
mkdir build
cd build

# Build the project
cmake ..
make -j$(nproc)

# Set Pangolin library path
PANGOLIN_LIB="/home/jake/calibration_w_eigan/third_party/Pangolin/install/lib"

# Update library cache
echo "Updating library cache..."
sudo ldconfig ${PANGOLIN_LIB}

# Set library path
export LD_LIBRARY_PATH="${PANGOLIN_LIB}:${LD_LIBRARY_PATH}"

# Verify library linking
echo "Checking library dependencies..."
ldd ./calibrate_3_cameras | grep pango

# Run the executable
./calibrate_3_cameras /home/jake/gripper_calib_data/detected_corners.csv