# Experiment Configuration

The configuration for each experiment (simulated or hardware) is specified
using a YAML file, which are typically stored in `upright_cmd/config/`.

## Including other YAML files

Often only a few configuration parameters change between experiments, so we
would like to be able to extend a general shared YAML file with only these few
differences. This is done using the `include` key, the value of which is block
sequence; each element of the sequence contains `package`, the name of the ROS
package, and `path`, the relative path to the YAML file from that package. The
element can also contain the `key` key, which specifies the key that included
YAML parameters should be placed in the overall configuration hierarchy.

For example, suppose I have my YAML file `mine.yaml` and I want to include the
parameters from `shared.yaml` located at `upright_cmd/config/shared.yaml`,
where `upright_cmd` is a ROS package. Then in `mine.yaml` (typically at the
top), I write:
```yaml
include:
  -
    package: upright_cmd
    path: config/shared.yaml
```

Now suppose that I also want to include some control parameters from the file
`upright_cmd/config/controller.yaml`, and I want these parameters to be nested
under the `controller` key. Then I can augment the `include` statement about
to
```yaml
include:
  -
    package: upright_cmd
    path: config/shared.yaml
  -
    key: controller
    package: upright_cmd
    path: config/controller.yaml
```

When one file includes another, any keys present in both take the value from
the file doing the including; in other words, the values in the included file
are overwritten. You can include as many files into another as desired.

## Parameters

The top-level keys for the upright project are
```yaml
controller  # controller parameters
simulation  # simulation parameters
logging     # how/where to log data (only used for simulation)
```

### Logging

Nested under `logging` are the following keys:
```yaml
timestep: float  # how often to record data (in seconds)
log_dir:  str    # the absolute path to directory in which to save data
```
These values are used by the `DataLogger` in
`upright_core/src/upright_core/logging.py`.

### Controller
Nested under `controller` are the following keys:
```yaml
gravity: list, length 3    # gravity vector

# upright uses OCS2's auto-differentiation + code generation to automatically
# compute gradients of costs and constraints
# set this to `true` to recompile each time, or `false` to skip this step.
# Only set to `true` if running the same controller setup repeatedly.
recompile_libraries: bool

# Enable extra debugging information. Currently, this is used to print and
# publish more information from the MRT node in
# `upright_ros_interface/src/mrt_node.cpp`.
debug: bool

# The solver to use. Currently only `SQP` (sequential quadratic programming)
# is supported.
solver_method: str

# settings for the model predictive controller
mpc:
  time_horizon: float, non-negative  # optimization time horizon
  debug_print: bool                  # set `true` to print extra information
  cold_start: bool                   # set `false` to warm start the solver at each control iteration

# settings for performing the forward rollout of the trajectory
rollout
```

### Simulation
