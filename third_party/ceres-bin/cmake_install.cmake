# Install script for directory: /home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/jake/calibration_w_eigan/third_party/ceres-bin/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/Ceres" TYPE FILE FILES
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/cmake/FindSuiteSparse.cmake"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/cmake/FindMETIS.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/jake/calibration_w_eigan/third_party/ceres-bin/internal/ceres/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/jake/calibration_w_eigan/third_party/ceres-bin/examples/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ceres" TYPE FILE FILES
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/autodiff_cost_function.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/autodiff_first_order_function.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/autodiff_manifold.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/c_api.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/ceres.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/conditioned_cost_function.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/constants.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/context.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/cost_function.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/cost_function_to_functor.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/covariance.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/crs_matrix.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/cubic_interpolation.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/dynamic_autodiff_cost_function.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/dynamic_cost_function.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/dynamic_cost_function_to_functor.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/dynamic_numeric_diff_cost_function.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/evaluation_callback.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/first_order_function.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/gradient_checker.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/gradient_problem.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/gradient_problem_solver.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/iteration_callback.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/jet.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/jet_fwd.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/line_manifold.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/loss_function.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/manifold.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/manifold_test_utils.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/normal_prior.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/numeric_diff_cost_function.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/numeric_diff_first_order_function.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/numeric_diff_options.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/ordered_groups.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/problem.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/product_manifold.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/rotation.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/sized_cost_function.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/solver.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/sphere_manifold.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/tiny_solver.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/tiny_solver_autodiff_function.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/tiny_solver_cost_function_adapter.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/types.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/version.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ceres/internal" TYPE FILE FILES
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/internal/array_selector.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/internal/autodiff.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/internal/disable_warnings.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/internal/eigen.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/internal/euler_angles.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/internal/fixed_array.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/internal/householder_vector.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/internal/integer_sequence_algorithm.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/internal/jet_traits.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/internal/line_parameterization.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/internal/memory.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/internal/numeric_diff.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/internal/parameter_dims.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/internal/port.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/internal/reenable_warnings.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/internal/sphere_manifold_functions.h"
    "/home/jake/calibration_w_eigan/third_party/ceres-solver-2.2.0/include/ceres/internal/variadic_evaluate.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/jake/calibration_w_eigan/third_party/ceres-bin/include/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/Ceres/CeresTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/Ceres/CeresTargets.cmake"
         "/home/jake/calibration_w_eigan/third_party/ceres-bin/CMakeFiles/Export/9a3bb6344a10c987f9c537d2a0e39364/CeresTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/Ceres/CeresTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/Ceres/CeresTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/Ceres" TYPE FILE FILES "/home/jake/calibration_w_eigan/third_party/ceres-bin/CMakeFiles/Export/9a3bb6344a10c987f9c537d2a0e39364/CeresTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/Ceres" TYPE FILE FILES "/home/jake/calibration_w_eigan/third_party/ceres-bin/CMakeFiles/Export/9a3bb6344a10c987f9c537d2a0e39364/CeresTargets-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/Ceres" TYPE FILE RENAME "CeresConfig.cmake" FILES "/home/jake/calibration_w_eigan/third_party/ceres-bin/CeresConfig-install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/Ceres" TYPE FILE FILES "/home/jake/calibration_w_eigan/third_party/ceres-bin/CeresConfigVersion.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/jake/calibration_w_eigan/third_party/ceres-bin/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
