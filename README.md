# Upright: Mobile Manipulator Object Balancing

Simulation and testing code for a mobile manipulator balancing objects on its
end effector. Simulator is Pybullet.

The code is designed to run on ROS Noetic. There is Docker image available in
the `docker/` directory if you are not running Ubuntu 20.04 with Noetic
natively.

## Contents
* `docker/`: Dockerfile and utility scripts to install and run things under ROS
  Noetic on Ubuntu 20.04.
* `upright_assets/`: URDF and mesh files.
* `upright_core/`: Core API for computing motion constraints required to
  balance objects. To be renamed to `upright_core`.
* `upright_control/`: Model predictive controller using the
  [OCS2](https://github.com/leggedrobotics/ocs2) framework. To be renamed to
  `upright_ctrl`.
* `upright_cmd`: Configuration and command scripts. Simulations and experiments
  are run from here.
* `upright_msgs`: Custom ROS messages.
* `upright_ros_interface`: Tools for ROS communication. These can be useful in
  simulation for multi-processing, or to support real hardware.
* `upright_sim/`: Simulation environments for balancing objects.

## Setup and Installation

Install required apt packages (TODO: this list is not exhaustive):
```
sudo apt install ros-noetic-eigenpy ros-noetic-hpp-fcl
```

TODO Python dependencies

Clone and build [pinocchio](https://github.com/stack-of-tasks/pinocchio) in a
separate folder outside of the catkin workspace. It can be built with catkin,
but I prefer not to because (1) if you ever clean and rebuild the workspace,
compiling pinocchio takes ages, and (2) I ran into an issue where it would
cause sourcing `devel/setup.bash` not to work properly (`ROS_PACKAGE_PATH`
wasn't set). Be sure to build it with hpp-fcl support (this can be done by
either editing the CMakeLists.txt or passing the compile option
`-DPINOCCHIO_WITH_HPP_FCL`), as well as the correct Python binary.

Clone dependencies into the `src` folder of your catkin workspace (I like to
put them in a subfolder called `tps` for "third-party software"):
* [OCS2](https://github.com/leggedrobotics/ocs2)
* [mobile_manipulation_central](https://github.com/utiasDSL/dsl__projects__mobile_manipulation_central)
* TODO more dependencies required for experiments

Clone this repo and build the workspace:
```
git clone https://github.com/utiasDSL/dsl__projects__tray_balance
catkin build
```

## Simulation

Simulation scripts are in `upright_cmd/scripts/sim`.

## Hardware

Interaction with hardware is done over ROS via mobile_manipulation_central.
