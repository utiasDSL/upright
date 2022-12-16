# Upright: Nonprehensile Object Transportation (with a Position-Controlled Mobile Manipulator)

Simulation and testing code for a mobile manipulator balancing objects on its
end effector. Simulator is Pybullet.

The code is designed to run on ROS Noetic. There is Docker image available in
the `docker/` directory if you are not running Ubuntu 20.04 with Noetic
natively.

## Contents
* `docker/`: Dockerfile and utility scripts to install and run things under ROS
  Noetic on Ubuntu 20.04.
* `upright_assets`: URDF and mesh files.
* `upright_core`: Core API for computing motion constraints required to
  balance objects.
* `upright_control`: Model predictive controller using the
  [OCS2](https://github.com/leggedrobotics/ocs2) framework.
* `upright_cmd`: Configuration and command scripts. Simulations and experiments
  are run from here.
* `upright_ros_interface`: Tools for ROS communication. These can be useful in
  simulation for multi-processing, or to support real hardware.
* `upright_sim`: Simulation environments for balancing objects.

## Setup and Installation

Install required apt packages:
```
sudo apt install ros-noetic-eigenpy ros-noetic-hpp-fcl
```

Clone and build [pinocchio](https://github.com/stack-of-tasks/pinocchio) in a
separate folder outside of the catkin workspace. It can be built with catkin,
but I prefer not to because (1) if you ever clean and rebuild the workspace,
compiling pinocchio takes ages, and (2) I ran into an issue where it would
cause sourcing `devel/setup.bash` not to work properly (`ROS_PACKAGE_PATH`
wasn't set). Be sure to build it with hpp-fcl support (this can be done by
either editing the CMakeLists.txt or passing the compile option
`-DPINOCCHIO_WITH_HPP_FCL`), as well as the correct Python binary. Follow the
installation directions
[here](https://stack-of-tasks.github.io/pinocchio/download.html), using the
cmake command:
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DPYTHON_EXECUTABLE=/usr/bin/python3 -DBUILD_WITH_COLLISION_SUPPORT=ON
```
Ensure that you also modify `$PYTHONPATH` to include the location of
pinocchio's Python bindings.

Clone catkin package dependencies into the `src` folder of your catkin
workspace (I like to put them in a subfolder called `tps` for "third-party
software"):
* Our custom fork of [OCS2](https://github.com/utiasDSL/ocs2). Install
  dependencies as listed
  [here](https://leggedrobotics.github.io/ocs2/installation.html) and then
  clone:
  ```
  git clone -b upright https://github.com/utiasDSL/ocs2
  ```
* [mobile_manipulation_central](https://github.com/utiasDSL/dsl__projects__mobile_manipulation_central)
  and its dependenices.

Clone this repo into the catkin workspace:
```
git clone https://github.com/utiasDSL/dsl__projects__tray_balance catkin_ws/src/upright
```

Install Python dependencies:
```
python3 -m pip install -r catkin_ws/src/upright/requirements.txt
```

Build the workspace:
```
catkin build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

## Simulation

Simulation scripts are in `upright_cmd/scripts/sim`.

## Hardware

Interaction with hardware is done over ROS via mobile_manipulation_central.
