cmake_minimum_required(VERSION 3.10)
project(calibrate_2_cameras)

# Paths to dependencies
set(PANGOLIN_DIR ${CMAKE_SOURCE_DIR}/third_party/Pangolin/install)
set(CERES_DIR ${CMAKE_SOURCE_DIR}/third_party/ceres-bin1/install)
set(EIGEN_DIR ${CMAKE_SOURCE_DIR}/third_party/eigen-3.4.0)

# Enable RPATH for finding shared libraries at runtime
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
# set(CMAKE_INSTALL_RPATH "${PANGOLIN_DIR}/lib")

# Make sure RPATH is added to the executable at build time
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Add both Pangolin and Ceres lib dirs to RPATH
set(CMAKE_INSTALL_RPATH "${PANGOLIN_DIR}/lib;${CERES_DIR}/lib")


# Executable
add_executable(calibrate_2_cameras calibrate_2_cameras.cpp)
add_executable(calibrate_3_cameras calibrate_3_cameras.cpp)
add_executable(vizualize_3_cameras_data vizualize_3_cameras_data.cpp)


set_target_properties(calibrate_2_cameras PROPERTIES
    BUILD_RPATH "${PANGOLIN_DIR}/lib;${CERES_DIR}/lib"
)
set_target_properties(calibrate_3_cameras PROPERTIES
    BUILD_RPATH "${PANGOLIN_DIR}/lib;${CERES_DIR}/lib"
)
set_target_properties(vizualize_3_cameras_data PROPERTIES
    BUILD_RPATH "${PANGOLIN_DIR}/lib;${CERES_DIR}/lib"
)


find_package(Threads REQUIRED)

# Includes
target_include_directories(calibrate_2_cameras PRIVATE
    ${PANGOLIN_DIR}/include
    ${CERES_DIR}/include
    ${EIGEN_DIR}
    ${CMAKE_SOURCE_DIR}/third_party
)
target_include_directories(calibrate_3_cameras PRIVATE
    ${PANGOLIN_DIR}/include
    ${CERES_DIR}/include
    ${EIGEN_DIR}
    ${CMAKE_SOURCE_DIR}/third_party
)
target_include_directories(vizualize_3_cameras_data PRIVATE
    ${PANGOLIN_DIR}/include
    ${CERES_DIR}/include
    ${EIGEN_DIR}
    ${CMAKE_SOURCE_DIR}/third_party
)

# Libraries
target_link_directories(calibrate_2_cameras PRIVATE
    ${PANGOLIN_DIR}/lib
    ${CERES_DIR}/lib
)
target_link_directories(calibrate_3_cameras PRIVATE
    ${PANGOLIN_DIR}/lib
    ${CERES_DIR}/lib
)
target_link_directories(vizualize_3_cameras_data PRIVATE
    ${PANGOLIN_DIR}/lib
    ${CERES_DIR}/lib
)

# Link libraries
target_link_libraries(calibrate_2_cameras
    pango_core
    pango_display
    pango_geometry
    pango_image
    pango_opengl
    pango_packetstream
    pango_plot
    pango_scene
    pango_tools
    pango_video
    pango_windowing
    pango_vars

    GL
    GLU
    glut
    GLEW

    ceres
    cholmod
    ccolamd
    colamd
    amd
    camd
    suitesparseconfig
    blas
    lapack
    glog
    gflags
    Threads::Threads
)
# Link libraries
target_link_libraries(calibrate_3_cameras
    pango_core
    pango_display
    pango_geometry
    pango_image
    pango_opengl
    pango_packetstream
    pango_plot
    pango_scene
    pango_tools
    pango_video
    pango_windowing
    pango_vars

    GL
    GLU
    glut
    GLEW

    ceres
    cholmod
    ccolamd
    colamd
    amd
    camd
    suitesparseconfig
    blas
    lapack
    glog
    gflags
    Threads::Threads
)
# Link libraries
target_link_libraries(vizualize_3_cameras_data
    pango_core
    pango_display
    pango_geometry
    pango_image
    pango_opengl
    pango_packetstream
    pango_plot
    pango_scene
    pango_tools
    pango_video
    pango_windowing
    pango_vars

    GL
    GLU
    glut
    GLEW

    ceres
    cholmod
    ccolamd
    colamd
    amd
    camd
    suitesparseconfig
    blas
    lapack
    glog
    gflags
    Threads::Threads
)


# C++ Standard
target_compile_features(calibrate_2_cameras PRIVATE cxx_std_17)
target_compile_features(calibrate_3_cameras PRIVATE cxx_std_17)
target_compile_features(vizualize_3_cameras_data PRIVATE cxx_std_17)
