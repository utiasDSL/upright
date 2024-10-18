cmake_minimum_required(VERSION 3.10.2)
project(upright_ros_interface)

add_compile_options(-std=c++14)

set(PYTHON_VERSION_STRING 3.8)

find_package(PythonLibs REQUIRED)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  upright_control
  std_msgs
  geometry_msgs
  pybind11_catkin
  mobile_manipulation_central
)

# hack to resolve pinocchio error, see
# <https://github.com/epfl-lasa/control-libraries/issues/120>
find_package(Boost REQUIRED COMPONENTS system)
add_compile_definitions(BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS)
add_compile_definitions(BOOST_MPL_LIMIT_LIST_SIZE=30)

catkin_python_setup()


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES upright_ros_interface
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${mobile_manipulation_central_INCLUDE_DIRS}
  ${upright_control_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(mpc_node src/mpc_node.cpp)
target_link_libraries(mpc_node
  ${catkin_LIBRARIES}
)

add_executable(mrt_node src/mrt_node.cpp)
target_link_libraries(mrt_node
  ${catkin_LIBRARIES}
)

add_executable(track_plan_node src/track_plan_node.cpp)
target_link_libraries(track_plan_node
  ${catkin_LIBRARIES}
)

add_executable(python_cpp src/python_cpp.cpp)
target_link_libraries(python_cpp ${PYTHON_LIBRARIES} ${upright_control_LIBRARIES})
