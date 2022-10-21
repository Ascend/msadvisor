# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(CompileDemo)

# Add definitions ENABLE_DVPP_INTERFACE to use dvpp api
add_definitions(-DENABLE_DVPP_INTERFACE)

# Check environment variable
if(NOT DEFINED ENV{ASCEND_HOME})
    message(FATAL_ERROR "please define environment variable:ASCEND_HOME")
endif()

# Find ascendbase
set(ASCEND_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ascendbase/src/Base)
get_filename_component(ASCEND_BASE_ABS_DIR ${ASCEND_BASE_DIR} ABSOLUTE)
file(GLOB_RECURSE ASCEND_BASE_SRC_FILES
    ${ASCEND_BASE_ABS_DIR}/CommandParser/*cpp
    ${ASCEND_BASE_ABS_DIR}/ConfigParser/*cpp
    ${ASCEND_BASE_ABS_DIR}/ErrorCode/*cpp
    ${ASCEND_BASE_ABS_DIR}/FileManager/*cpp
    ${ASCEND_BASE_ABS_DIR}/Log/*cpp
    ${ASCEND_BASE_ABS_DIR}/AsynLog/*cpp
)

# Compile options
add_compile_options(-std=c++11 -fPIE -fstack-protector-all -Wall)

# Skip build rpath
set(CMAKE_SKIP_BUILD_RPATH True)

# Set output directory
set(PROJECT_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR}/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SRC_ROOT}/dist)

# Set include directory and library directory
set(ACL_INC_DIR $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/$ENV{ARCH_PATTERN}/include)
set(ACL_LIB_DIR $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/$ENV{ARCH_PATTERN}/lib64/stub)

# Header path
include_directories(${ACL_INC_DIR})
include_directories(${ASCEND_BASE_DIR})
include_directories(${ASCEND_BASE_DIR}/Framework)

# add host lib path
link_directories(${ACL_LIB_DIR})

add_executable(main ${SRC_FILES} ${ASCEND_BASE_SRC_FILES} ${PROJECT_SRC_ROOT}/main.cpp)

target_link_libraries(main ascendcl acl_dvpp pthread -Wl,-z,relro,-z,now,-z,noexecstack -pie -s)