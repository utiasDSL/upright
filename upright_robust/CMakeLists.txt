cmake_minimum_required(VERSION 3.0.2)
project(upright_robust)

add_compile_options(-std=c++14)
find_package(catkin REQUIRED COMPONENTS pybind11_catkin upright_core)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)


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
  INCLUDE_DIRS include
  # LIBRARIES ${PROJECT_NAME}
  # CATKIN_DEPENDS pybind11_catkin
  # DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${upright_core_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

pybind11_add_module(bindings SHARED
  src/pybindings.cpp
)

catkin_python_setup()

#############
## Install ##
#############

install(TARGETS bindings
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
set_target_properties(bindings
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
