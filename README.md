# Upright

Code for solving the waiter's problem with model predictive control on a mobile
manipulator. Accompanies [this paper](https://arxiv.org/abs/2305.17484).

The code is designed to run on ROS Noetic. There is Docker image available in
the `docker/` directory if you are not running Ubuntu 20.04 with Noetic
natively. For experiments on real hardware, it is highly recommended to use a
real-time system like Linux with the PREEMPT_RT patch.

## Contents
* `upright_assets`: URDF and mesh files.
* `upright_core`: Core API for computing motion constraints required to
  balance objects.
* `upright_control`: Model predictive controller using the
  [OCS2](https://github.com/leggedrobotics/ocs2) framework.
* `upright_cmd`: Configuration and command scripts. Simulations and experiments
  are run from here, as well as other smaller scripts and tools.
* `upright_ros_interface`: Tools for ROS communication. These can be useful in
  simulation for multi-processing, or to support real hardware.
* `upright_sim`: (PyBullet) simulation environments for balancing objects.

## Setup and Installation

Install eigenpy (a Pinocchio dependency) from apt:
```
sudo apt install ros-noetic-eigenpy
```

Clone and build [hpp-fcl](https://github.com/humanoid-path-planner/hpp-fcl)
from source. Unfortunately the master branch and the version available on apt
under the name `ros-noetic-hpp-fcl` has a
[bug](https://github.com/humanoid-path-planner/hpp-fcl/issues/344) which causes
contact normals to contain NaN values. However, that is fixed on the more
recent `hppfcl3x` branch. Do:
```
# get the code with the bug fixed
git clone https://github.com/humanoid-path-planner/hpp-fcl
cd hpp-fcl
git checkout hppfcl3x

# build and install it
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
sudo make install
```

Clone and build [Pinocchio](https://github.com/stack-of-tasks/pinocchio) in a
separate folder outside of the catkin workspace. It can be built with catkin,
but I prefer not to because (1) if you ever clean and rebuild the workspace,
compiling Pinocchio takes ages, and (2) I ran into an issue where it would
cause sourcing `devel/setup.bash` not to work properly (`ROS_PACKAGE_PATH`
wasn't set). Follow the installation directions
[here](https://stack-of-tasks.github.io/pinocchio/download.html) (under the
"Build from Source" tab), using the cmake command:
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DPYTHON_EXECUTABLE=/usr/bin/python3 -DBUILD_WITH_COLLISION_SUPPORT=ON
```
Ensure that you also modify `$PYTHONPATH` to include the location of
Pinocchio's Python bindings.

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

## Simulation Experiments

Simulation scripts are in `upright_cmd/scripts/simulations`. For example, to
run a simulation without ROS, do something like:
```
upright_cmd/scripts/simulations
./mpc_sim --config <path to yaml file>
```
All experiments, whether simulated or real, are specified by configuration
files in the YAML format, which are stored under `upright_cmd/config`.

## Hardware Experiments

Interaction with hardware is done over ROS via
[mobile_manipulation_central](https://github.com/utiasDSL/dsl__projects__mobile_manipulation_central).
So far we have targetted an omnidirectional mobile manipulator consisting of a
Ridgeback mobile base and a UR10 manipulator arm (collectively named the
"Thing"). The general flow of experiments is to connect to the robot, and run
```
roslaunch mobile_manipulation_central thing.launch
```
Then in another terminal run
```
roslaunch upright_ros_interface mpc_mrt.launch config:=<path to yaml file>
```
You may wish to record the results in a bag file using the
`upright_cmd/scripts/record.py` script, which is just a wrapper around `rosbag`.

## Tests

Some packages contain tests. Python tests use [pytest](https://pytest.org/).
Run `pytest .` inside a package's `tests` directory to run the Python tests.

## Citation

If you find this work useful, feel free to cite the accompanying
[paper](https://arxiv.org/abs/2305.17484):
```
@misc{heins2023upright,
      title={Keep it Upright: Model Predictive Control for Nonprehensile Object Transportation with Obstacle Avoidance on a Mobile Manipulator}, 
      author={Adam Heins and Angela P. Schoellig},
      year={2023},
      eprint={2305.17484},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## License

MIT (see the LICENSE file).
