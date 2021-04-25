cmake_minimum_required(VERSION 2.8)

project( asmirnov_demo )

# OpenCV 

find_package( OpenCV REQUIRED )

# set(OpenCV_INCLUDE_DIRS "/home/asmirnov/OV_ARM_package/opencv/include")
include_directories( ${OpenCV_INCLUDE_DIRS} )

# InferenceEngine

find_package(InferenceEngine)
include_directories(${InferenceEngine_INCLUDE_DIRS})

add_executable( asmirnov_demo src.cpp )
target_link_libraries( asmirnov_demo ${OpenCV_LIBS}  ${InferenceEngine_LIBRARIES})