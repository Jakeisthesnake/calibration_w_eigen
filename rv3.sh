#!/bin/bash
cd build
# Set correct library path
export LD_LIBRARY_PATH=/home/jake/calibration_w_eigen/third_party/Pangolin/install/lib:$LD_LIBRARY_PATH

# Run the executable
# ./build/vizualize_3_cameras_data /home/jake/gripper_calib_data/detected_corner_frac.csv /home/jake/calibration_w_eigen/calibration_output.json
# ./vizualize_3_cameras_data /home/jake/calibration_w_python/synthetic_data_3_cams.csv /home/jake/calibration_w_python/synthetic_calibration.json
# ./vizualize_3_cameras_data /home/jake/calibration_w_python/synthetic_data_3_cams.csv /home/jake/calibration_w_eigen/calibration_result_final.json
# ./vizualize_3_cameras_data /home/jake/calibration_w_python/synthetic_data_3_cams.csv /home/jake/calibration_w_eigen/calibration_result_initial.json
# ./vizualize_3_cameras_data /home/jake/calibration_w_python/synthetic_data_3_cams.csv /home/jake/calibration_w_eigen/calibration_result_initial_homg_all.json
./vizualize_3_cameras_data /home/jake/calibration_w_python/synthetic_data_3_cams.csv /home/jake/calibration_w_eigen/calibration_post_processing.json
# ./vizualize_3_cameras_data /home/jake/calibration_w_python/synthetic_data_3_cams.csv /home/jake/calibration_w_eigen/calib_iter_0.json
# ./vizualize_3_cameras_data /home/jake/calibration_w_python/synthetic_data_3_cams_frame_14.csv /home/jake/calibration_w_eigen/calib_iter_3.json