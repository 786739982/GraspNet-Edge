cmake_minimum_required(VERSION 3.4...3.18)
project(airbot)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Debug)

find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED)
find_package(airbot_play REQUIRED)
find_package(assimp REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization system thread filesystem)
find_package(Eigen3 REQUIRED)
# find_package(kdl_parser REQUIRED)
find_package(Threads)
find_package(orocos_kdl REQUIRED)
find_package(Curses REQUIRED)
find_package(spdlog REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${orocos_kdl_INCLUDE_DIRS}
)

set(DEPENDENCIES
  airbot_play
  ${NCURSES_LIBRARIES}
  Eigen3::Eigen
  ${orocos_kdl_LIBRARIES}
  # kdl_parser
  spdlog::spdlog
  nlohmann_json
  argparse
  ${OpenCV_LIBS}
)

# FIXME: This is a hack to get the correct include directory for arm_control
message(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/airbot)

add_subdirectory(pybind11)
pybind11_add_module(airbot src/main.cpp)
target_include_directories(airbot PRIVATE ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/airbot)
target_link_libraries(airbot PRIVATE
  Python::Python
  ${DEPENDENCIES}
)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(airbot PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
