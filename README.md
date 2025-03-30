<p align="center">
<img src="https://static.adamheins.com//upright/logo.svg" alt="Robot waiter." width="15%"/>
</p>

Code for solving the *waiter's problem* with online (MPC) or offline planning on
a mobile manipulator. The waiter's problem refers to moving while keeping
objects balanced on a tray-like end effector (like a waiter in a restaurant),
which is an example of *nonprehensile* manipulation.

The code in this repository accompanies two papers:
* [Keep It Upright: Model Predictive Control for Nonprehensile Object Transportation With Obstacle Avoidance on a Mobile Manipulator](https://arxiv.org/abs/2305.17484),
* [Robust Nonprehensile Object Transportation with Uncertain Inertial Parameters](https://arxiv.org/abs/2411.07079).

A full video from the first paper can be found
[here](http://tiny.cc/keep-it-upright).

Some examples include reacting to sudden changes in the environment:

![Sudden obstacle avoidance](https://static.adamheins.com/upright/sudden.gif)

and avoiding dynamic obstacles like thrown balls:

![Dynamic obstacle avoidance](https://static.adamheins.com/upright/dodge.gif)

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
* `upright_robust`: Robust planning for balancing objects with uncertain inertial parameters. See [here](upright_robust/README.md) for more details.

## Setup and Installation

The code is designed to run on ROS Noetic under Ubuntu 20.04. For experiments
on real hardware, it is highly recommended to use a real-time system like Linux
with the PREEMPT_RT patch.

First, follow the instructions to setup the
[mobile_manipulation_central](https://github.com/utiasDSL/mobile_manipulation_central)
repo in your catkin workspace.

Next, clone our custom fork of [OCS2](https://github.com/utiasDSL/ocs2):
```
git clone -b upright https://github.com/utiasDSL/ocs2
```
Install dependencies as listed
[here](https://leggedrobotics.github.io/ocs2/installation.html).

Now you can clone this repo into the catkin workspace:
```
git clone https://github.com/utiasDSL/upright catkin_ws/src/upright
```

Install Python dependencies:
```
python3 -m pip install -r catkin_ws/src/upright/requirements.txt
```

There are many OCS2 packages that can be skipped. You can use the catkin
[config.yaml](https://github.com/utiasDSL/mobile_manipulation_central/blob/main/catkin/config/yaml)
file; place it under `catkin_ws/.catkin_tools/profiles/default/`.

Finally, build the workspace:
```
catkin build
```


## Simulation Experiments

Simulation scripts are in `upright_cmd/scripts/simulations`. For example, to
run a simulation without ROS, do something like:
```
cd upright_cmd/scripts/simulations
./mpc_sim --config <path to yaml file>

# for example
# thing_demo.yaml uses the entire mobile manipulator
# ur10_demo.yaml uses only the arm
./mpc_sim --config $(rospack find upright_cmd)/config/demos/thing_demo.yaml
```
All experiments, whether simulated or real, are specified by configuration
files in the YAML format, which are stored under `upright_cmd/config`. Configuration parameters are documented [here](docs/configuration.md).

Note that once the simulation is setup (this can take some time when
re-compiling auto-differentiated libraries), you will be dropped into an
IPython shell, which should look something like:
```
In [1]:
```
This allows you to inspect any variables of interest. If you just want to start
executing the simulated trajectory, type `exit` and press Enter to exit the
shell and continue.

## Hardware Experiments

Interaction with hardware is done over ROS via
[mobile_manipulation_central](https://github.com/utiasDSL/mobile_manipulation_central).
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

## Citations

If you find this work useful, feel free to cite (one of) the accompanying
papers.

The [original paper](https://doi.org/10.1109/LRA.2023.3324520) on fast MPC for
the waiter's problem is:
```
@article{heins2023upright,
  title = {Keep It Upright: Model Predictive Control for Nonprehensile Object Transportation With Obstacle Avoidance on a Mobile Manipulator},
  author = {Adam Heins and Angela P. Schoellig},
  journal = {{IEEE Robotics and Automation Letters}},
  number = {12},
  volume = {8},
  pages = {7986--7993},
  doi = {10.1109/LRA.2023.3324520},
  year = {2023},
}
```

The [follow-up paper](https://arxiv.org/abs/2411.07079) on robust planning for
the waiter's problem under inertial parameter uncertainty:
```
@article{heins2025robust,
  title = {Robust Nonprehensile Object Transportation with Uncertain Inertial Parameters},
  author = {Adam Heins and Angela P. Schoellig},
  journal = {{IEEE Robotics and Automation Letters}},
  number = {5},
  volume = {10},
  pages = {4492--4499},
  doi = {10.1109/LRA.2025.3551067},
  year = {2025},
}
```

## License

MIT (see the LICENSE file).
