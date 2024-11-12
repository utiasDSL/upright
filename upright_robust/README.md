# Robust Nonprehensile Object Transportation

Nonprehensile object transportation that is robust to uncertainty in the
inertial parameters (mass, center of mass, inertia matrix). We use the same
optimal control problem formulation as the original work in this repository,
but instead solve the problem once offline with a long time horizon. The plan
is then tracked online. More details can be found in the
[paper](https://arxiv.org/abs/2411.07079).

## Simulations

* Run the simulation experiments with `scripts/planning_sim_loop.py`.
* Process the results (i.e., verify robustness) using
  `scripts/process_sim_runs.py`.

## Experiments

The node for running the planner online with the real robot lives in the
`upright_ros_interface` package, to keep all of the ROS infrastructure
contained in one place. To run it, do
```
roslaunch upright_ros_interface track_plan.launch config:=<path to yaml file>
```
The plan is generated and then tracked online the robot.
