#!/bin/bash

# Set correct library path
export LD_LIBRARY_PATH=/home/jake/calibration_w_eigen/third_party/Pangolin/install/lib:$LD_LIBRARY_PATH

# Run the executable
./build/vizualize_3_cameras_data /home/jake/gripper_calib_data/detected_corner_frac.csv /home/jake/calibration_w_eigen/calibration_output.json
# ./build/vizualize_3_cameras_data /home/jake/calibration_w_python/synthetic_data_3.csv /home/jake/calibration_w_python/synthetic_calibration.json