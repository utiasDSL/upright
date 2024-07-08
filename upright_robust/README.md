# Robust Nonprehensile Object Transportation

Nonprehensile object transportation that is robust to uncertainty in the
inertial parameters (mass, center of mass, inertia matrix).

## Configuration

Tilting type:

* **tray**: rotate the tray so its normal vector is aligned with total
  acceleration, neglecting the object CoMs, but constraints are still enforced
  to avoid dropping them
* **tray_only**: same as tray but without any balancing constraints
* **full**: take all objects into accounting when tilting/rotating
* **flat**: keep the tray flat

Constraint type:

* **nominal**: nominal balancing constraints based on some guess of the
  inertial parameters, does not account for uncertainty
* **robust**: balancing constraints robust to uncertainty in the inertial
  parameters
* **approx_robust**: instead of solving the full robust QP, solve the nominal
  one and then just scale the resulting acceleration to satisfy the robust
  balancing constraints. This is useful when there are many (3+) objects and
  the QP becomes too slow to solve at real-time rates.

If `reactive.face_form=true`, the face form of the robust constraints is used
(rather than the original span form).

To remove all balancing constraints, set `balancing.enabled=false`.

## SDP relaxation

The work on SDP relaxations can be found in `scripts/relaxation`. The
relaxations integrated with the config files can be found in
`sdp_relaxation.py`. Use `--verify` to verify that a single approximate inertia
value with always hold, or use `--elimination` to detect redundant constraints
that can be eliminated.

## Simulation

To run the simulation normally use `scripts/simulation.py`.

To run the simulation using ROS to talk to the controller, ensure ROS master is
running. Then run the simulation using `scripts/ros_simulation.py` followed by
the controller using `scripts/ros_controller.py`.

## Hardware

When first starting out, one needs to build up task complexity gradually to
ensure things work as expected:
1. Use `nominal_flat/short.yaml` after having modified the goal waypoint to
   zero, in order to remain stationary.
2. Use `nominal_full/short.yaml`, again with stationary trajectory, to ensure
   balancing constraints are okay with rotation capabilities.
3. Repeat 1 and 2, slowly increasing the waypoint distance to 2 meters.

Other notes:
* The KF may need tuning.
* Using an actual pre-planned trajectory does not make too much sense, since we
  do not know how fast the object can actually travel.
