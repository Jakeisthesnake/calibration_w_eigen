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
# NEW INITIALIZATION SCHEMA (requires -intrinsicsfile):
./calibrate_3_cameras -datafile /home/jake/calibration_w_python/synthetic_data_3_cams_frame_1.csv -intrinsicsfile /home/jake/calibration_w_eigen/intrinsics_default.json -perframeflags /home/jake/calibration_w_eigen/per_frame_flags_extrinsics_only.json -globalflags /home/jake/calibration_w_eigen/global_flags_default.json
# ./calibrate_3_cameras -datafile /home/jake/calibration_w_python/synthetic_data_3_cams.csv -intrinsicsfile /home/jake/calibration_w_eigen/intrinsics_default.json -extrinsicsfile /home/jake/calibration_w_eigen/extrinsics_default.json -perframeflags /home/jake/calibration_w_eigen/per_frame_flags_default.json -globalflags /home/jake/calibration_w_eigen/global_flags_default.json

# OLD EXAMPLES (commented out):
# ./calibrate_3_cameras -datafile /home/jake/gripper_calib_data/detected_corner_frac.csv
# ./calibrate_3_cameras -datafile /home/jake/calibration_w_python/synthetic_data_3_cams.csv
# ./calibrate_3_cameras -datafile /home/jake/calibration_w_python/synthetic_data_3_cams_frame_1.csv -intrinsicsfile /home/jake/calibration_w_eigen/intrinsics_default.json -perframeflags /home/jake/calibration_w_eigen/per_frame_flags_extrinsics_only.json -globalflags /home/jake/calibration_w_eigen/global_flags_default.json
# ./calibrate_3_cameras -datafile /home/jake/calibration_w_python/synthetic_data_3_cams.csv -calibrationfile /home/jake/calibration_w_python/synthetic_calibration.json
# ./calibrate_3_cameras -datafile /home/jake/calibration_w_python/synthetic_data_3_cams.csv -calibrationfile /home/jake/calibration_w_eigen/calib_23_saved.json
# ./calibrate_3_cameras -datafile /home/jake/calibration_w_python/synthetic_data_3_cams.csv -calibrationfile /home/jake/calibration_w_eigen/calibration_result_initial_homg_all.json -intrinsicsfile /home/jake/calibration_w_eigen/intrinsics_default.json -extrinsicsfile /home/jake/calibration_w_eigen/extrinsics_default.json -perframeflags /home/jake/calibration_w_eigen/per_frame_flags_default.json -globalflags /home/jake/calibration_w_eigen/global_flags_default.json
# ./calibrate_3_cameras -datafile /home/jake/calibration_w_python/synthetic_data_3_cams_frame_14.csv -calibrationfile /home/jake/calibration_w_eigen/calibration_result_initial_homg_frame_14.json -intrinsicsfile /home/jake/calibration_w_eigen/intrinsics_default.json -extrinsicsfile /home/jake/calibration_w_eigen/extrinsics_default.json
# ./calibrate_3_cameras -datafile /home/jake/calibration_w_python/synthetic_data_3_cams_frame_8.csv -calibrationfile /home/jake/calibration_w_eigen/calibration_result_initial_homg_frame_8.json -intrinsicsfile /home/jake/calibration_w_eigen/intrinsics_default.json -extrinsicsfile /home/jake/calibration_w_eigen/extrinsics_default.json
