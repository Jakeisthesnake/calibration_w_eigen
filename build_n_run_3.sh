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
# ./calibrate_3_cameras /home/jake/gripper_calib_data/detected_corner_frac.csv
# ./calibrate_3_cameras /home/jake/calibration_w_python/synthetic_data_3_cams.csv
# ./calibrate_3_cameras /home/jake/calibration_w_python/synthetic_data_3_cams.csv /home/jake/calibration_w_python/synthetic_calibration.json
# ./calibrate_3_cameras /home/jake/calibration_w_python/synthetic_data_3_cams.csv /home/jake/calibration_w_eigen/calib_23_saved.json
./calibrate_3_cameras /home/jake/calibration_w_python/synthetic_data_3_cams.csv /home/jake/calibration_w_eigen/calibration_result_initial_homg_all.json