cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolo11_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_RKNN_YOLO11_DEMO "Build RKNN YOLOv10 Demo" ON)

# Check release or debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
endif()

# Set default optimization level for Release build
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Set the installation path to a local directory
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install/rknn_yolov10_demo_Linux")

# RK platform specific options
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(STATUS "Target platform: Linux")
    add_definitions(-DRK3588)
    
    # Include directories for RK3588
    set(RGA_INCLUDE
        ${CMAKE_SOURCE_DIR}/include/3rdparty/rga/RK3588/include)
    
    # Library directories for RK3588
    set(LIB_ARCH
        ${CMAKE_SOURCE_DIR}/include/3rdparty/rga/RK3588/lib/Linux/aarch64)
    
    # Set RKNN runtime library
    set(RKNN_RT_LIB
        ${CMAKE_SOURCE_DIR}/include/librknnrt.so)
    
    # Set RGA library
    set(RGA_LIB
        rga)
endif()

# Find OpenCV package
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Set installation RPATH
set(CMAKE_INSTALL_RPATH "lib")

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${RGA_INCLUDE}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src  # Add src directory to include path
)

# Link directories
link_directories(
    ${LIB_ARCH}
)

# Add the rknn_yolov10_demo executable
add_executable(rknn_yolov10_demo
    src/main.cc
    src/postprocess.cc
    src/yolov10.cc
    src/rkYolov10.cc
    src/file_utils.c
    src/image_utils.c
    src/rga_func.c
)

# Link against required libraries
target_link_libraries(rknn_yolov10_demo
    ${RKNN_RT_LIB}
    ${OpenCV_LIBS}
    ${RGA_LIB}
    dl
    pthread
)

# Install targets
install(TARGETS rknn_yolov10_demo DESTINATION .)

# Install libraries
install(FILES ${RKNN_RT_LIB} DESTINATION lib)
install(FILES ${LIB_ARCH}/librga.so DESTINATION lib)

# Install model files if they exist
if(EXISTS ${CMAKE_SOURCE_DIR}/model)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/model/ DESTINATION model)
endif()
