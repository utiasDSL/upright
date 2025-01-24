# Robust Nonprehensile Object Transportation

Nonprehensile object transportation that is robust to uncertainty in the
inertial parameters (mass, center of mass, inertia matrix). We use the same
optimal control problem formulation as the original work in this repository,
but instead solve the problem once offline with a long time horizon. The plan
is then tracked online. More details can be found in the
[paper](https://arxiv.org/abs/2411.07079).

## Simulations

Run the simulation experiments with `scripts/planning_sim_loop.py`. For
example, to collect the data with the box of height 30cm using the proposed
robust constraints, do:
```
./planning_sim_loop.py --config $(rospack find upright_robust)/config/demos/sim.yaml --height 30 --com robust
```
To see the simulation visually, use the `--gui` flag. To log the results, use
`--log`.

After running the simulations, they can be processed (e.g., to verify
robustness) using `scripts/process_sim_runs.py`. You need to provide the
directory where the simulation results were logged.

## Experiments

The node for running the planner online with the real robot lives in the
`upright_ros_interface` package, to keep all of the ROS infrastructure
contained in one place. To run it, do
```
roslaunch upright_ros_interface track_plan.launch config:=<path to yaml file>
```
The plan is generated and then tracked online the robot.
