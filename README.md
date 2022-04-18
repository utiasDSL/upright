# Mobile Manipulator Object Balancing

Simulation and testing code for a mobile manipulator balancing objects on its
end effector. Simulator is Pybullet.

The code is designed to run on ROS Noetic. There is Docker image available in
the `docker/` directory if you are not running Ubuntu 20.04 with Noetic
natively.

## Contents
* `docker/`: Dockerfile and utility scripts to install and run things under ROS
  Noetic on Ubuntu 20.04.
* `tray_balance_assets/`: URDF and mesh files.
* `tray_balance_constraints/`: API for computing motion constraints required to
  balance objects.
* `tray_balance_sim/`: Simulation environments for balancing objects.
* `tray_balance_ocs2/`: Model predictive controller using the
  [OCS2](https://github.com/leggedrobotics/ocs2) framework.

