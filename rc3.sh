#!/bin/bash

# Run calibrate 3 cameras

# Clean and create build directory
# rm -rf build
# mkdir build
# cd build

# Build the project
# cmake ..
# make -j$(nproc)

# Set correct library path
export LD_LIBRARY_PATH=/home/jake/calibration_w_eigen/third_party/Pangolin/install/lib:$LD_LIBRARY_PATH

# Run the executable
#gripper data, detected corner frac only
# ./calibrate_3_cameras -datafile /home/jake/gripper_calib_data/detected_corner_frac.csv

#synthetic data_3_cams, synthetic calibration only
./build/calibrate_3_cameras -datafile /home/jake/calibration_w_python/synthetic_data_3_cams.csv 

#synthetic data_3_cams, synthetic calibration, intrinsics default, extrinsics default
# ./build/calibrate_3_cameras -datafile /home/jake/calibration_w_python/synthetic_data_3_cams.csv -calibrationfile /home/jake/calibration_w_python/synthetic_calibration.json -intrinsicsfile /home/jake/calibration_w_eigen/intrinsics_default.json -extrinsicsfile /home/jake/calibration_w_eigen/extrinsics_default.json

#synthetic data_3_cams, synthetic calibration, intrinsics default, extrinsics default
# ./build/calibrate_3_cameras -datafile /home/jake/calibration_w_python/synthetic_data_3_cams.csv -calibrationfile /home/jake/calibration_w_python/synthetic_calibration.json -intrinsicsfile /home/jake/calibration_w_eigen/intrinsics_default.json -extrinsicsfile /home/jake/calibration_w_eigen/extrinsics_default.json

#synthetic data_3_cams, initial homg all, intrinsics default, extrinsics default
# ./build/calibrate_3_cameras -datafile /home/jake/calibration_w_python/synthetic_data_3_cams.csv -calibrationfile /home/jake/calibration_w_eigen/calibration_result_initial_homg_all.json -intrinsicsfile /home/jake/calibration_w_eigen/intrinsics_default.json -extrinsicsfile /home/jake/calibration_w_eigen/extrinsics_default.json


# ./build/calibrate_3_cameras -datafile /home/jake/calibration_w_python/synthetic_data_3_cams.csv -calibrationfile /home/jake/calibration_w_eigen/calibration_result_initial_homg_all.json -intrinsicsfile /home/jake/calibration_w_eigen/intrinsics_default.json -extrinsicsfile /home/jake/calibration_w_eigen/extrinsics_default.json -perframeflags /home/jake/calibration_w_eigen/per_frame_flags_default.json -globalflags /home/jake/calibration_w_eigen/global_flags_default.json

#synthetic data_3_cams only
# ./build/calibrate_3_cameras -datafile /home/jake/calibration_w_python/synthetic_data_3_cams.csv

#synthetic data_3_cams_frame_14, initial homg frame 14, intrinsics default, extrinsics default
# ./build/calibrate_3_cameras -datafile /home/jake/calibration_w_python/synthetic_data_3_cams_frame_14.csv -calibrationfile /home/jake/calibration_w_eigen/calibration_result_initial_homg_frame_14.json -intrinsicsfile /home/jake/calibration_w_eigen/intrinsics_default.json -extrinsicsfile /home/jake/calibration_w_eigen/extrinsics_default.json

#synthetic data_3_cams_frame_8, initial homg frame 8, intrinsics default, extrinsics default
# ./build/calibrate_3_cameras -datafile /home/jake/calibration_w_python/synthetic_data_3_cams_frame_8.csv -calibrationfile /home/jake/calibration_w_eigen/calibration_result_initial_homg_frame_8.json -intrinsicsfile /home/jake/calibration_w_eigen/intrinsics_default.json -extrinsicsfile /home/jake/calibration_w_eigen/extrinsics_default.json




