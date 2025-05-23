#!/bin/bash
set -e

# Target SOC and compiler settings
GCC_COMPILER=aarch64-linux-gnu
export LD_LIBRARY_PATH=${TOOL_CHAIN}/lib64:$LD_LIBRARY_PATH
export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++
ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

# Build directory
BUILD_DIR=${ROOT_PWD}/build/build_linux_aarch64
if [ ! -d "${BUILD_DIR}" ]; then
  mkdir -p ${BUILD_DIR}
fi

# Navigate to build directory and run CMake
cd ${BUILD_DIR}
cmake ../.. -DCMAKE_SYSTEM_NAME=Linux -DTARGET_SOC=RK3588

# Build with multiple cores for speed
make -j$(nproc)

# Install to the destination directory
make install

cd -

# Run the YOLOv10 demo with webcam
echo "Build completed. To run the YOLO v10 demo, use:"
echo "cd install/rknn_yolov10_demo_Linux/ && ./rknn_yolov10_demo ./model/RK3588/yolo10s-640-640.rknn webcam:0"