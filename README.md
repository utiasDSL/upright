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
* `tray_balance_constraints/`: Core API for computing motion constraints required to
  balance objects. To be renamed to `upright_core`.
* `tray_balance_ocs2/`: Model predictive controller using the
  [OCS2](https://github.com/leggedrobotics/ocs2) framework. To be renamed to
  `upright_ctrl`.
* `upright_cmd`: Configuration and command scripts. Simulations and experiments
  are run from here.
* `upright_msgs`: Custom ROS messages.
* `upright_ros_interface`: Tools for ROS communication. These can be useful in
  simulation for multi-processing, or to support real hardware.
* `upright_sim/`: Simulation environments for balancing objects.

## Setup and Installation

First, clone the repo and required dependencies into a catkin workspace. Build
the workspace:
```
catkin build
```
Generate the required URDFs:
```
cd upright_assets/thing
./compile_xacro.sh
```

## Simulation

## Hardware

### UR10
Following
[this](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver#prepare-the-ros-pc),
make sure you've extracted the factory calibration information at some point:
```
roslaunch upright_ros_interface ur10_calibration.launch
```
Then start the driver:
```
roslaunch upright_ros_interface ur10.launch
```
Finally, start the program on the Polyscope interface. The driver should print
that it has connected to the robot. The robot is now ready to accept commands.

### Robotiq Gripper
```bash
# connect to the gripper
roslaunch upright_ros_interface robotiq.launch

# Open the gripper if <cmd> is "o", else close it, after waiting for <delay>
# seconds.
rosrun upright_ros_interface gripper.py <cmd> <delay>
```
