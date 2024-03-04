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
gravity: list of float, length 3    # gravity vector

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

# settings for sytem state estimation
# estimation is done with a Kalman filter
# noise is assumed isotropic, so variance can be represented with a single scalar
estimation:
  robot_init_variance: float, non-negative         # initial pose variance
  robot_process_variance: float, non-negative      # process variance
  robot_measurement_variance: float, non-negative  # measurement variance

# settings for low-level reference tracking
tracking:
  rate: int, non-negative                      # controller frequency [Hz]
  min_policy_update_time: float, non-negative  # don't switch the MPC policy more often than this [s]

  # state feedback gains
  # these should be set to zero if sqp.use_feedback_policy = true, since the
  # controller computes its own optimal feedback policy in that case
  kp, kv, ka: float, non-negative

  # for each of these options, if it is `true`, then the controller with stop
  # when the state, input, or EE position limits are violated, respectively
  enforce_state_limits: bool
  enforce_input_limits: bool
  enforce_ee_position_limits: bool

  # margins for violation of the state, input, and EE position bounds
  state_violation_margin: float, non-negative
  input_violation_margin: float, non-negative
  ee_position_violation_margin: float, non-negative

  # set to `true` when doing projectile experiments
  use_projective: bool

# settings for the sequential quadratic programming solver
sqp:
  dt: float, non-negative            # time step of the optimized trajectory [s]
  sqp_iteration: int, positive       # max number of SQP iterations per solve
  init_sqp_iteration: int, positive  # max number of SQP iterations during first solve

  # convergence parameters
  delta_tol: float, non-negative
  cost_tol: float, non-negative

  # set `true` for MPC to compute a linear feedback policy
  use_feedback_policy: bool

  # set true to project the state and input onto the lower-dimensional space
  # defined by the affine inequality constraints
  project_state_input_equality_constraints: bool

  # print solver information
  print_solver_status: bool
  print_solver_statistics: bool
  print_line_search: bool

  # settings for the HPIPM QP solver used to solve each QP
  hpipm:
    iter_max: int, positive  # max number of iterations
    warm_start: bool         # set `true` to warm start the solver
    slacks:                  # slack variable settings
      enabled: bool          # enable slack variables

# settings for the balancing constraints
balancing:
  enabled: bool  # set `true` to enable the balancing constraints

  # name of the arrangement of objects being balanced
  arrangement: str

  # `hard` for hard constraints; `soft` to enforce constraints via cost penalties
  # soft constraints have not been extensively tested
  constraint_type: hard | soft

  # set `true` to use contact-forced based constraints, or `false` to use
  # constraints based on the zero-moment point and limit surface
  # the latter constraints are less flexible and are no longer fully supported
  use_force_constraints: bool

  # weight on the contact forces in the objective function
  force_weight: float, non-negative

# settings for the inertial alignment method, an alternative to the balancing
# constraints
# inertial alignment tries to tilt the tray so that its normal is always
# aligned opposite to the gravito-inertial acceleration
inertial_alignment:
  # set `true` to add inertial alignment as a cost
  cost_enabled: bool

  # set `true` to add inertial alignment as a constraint
  # should not be used in conjunction with `cost_enabled`
  constraint_enabled: bool

  # take the angular acceleration of the specified `com` into account when
  # computing alignment
  use_angular_acceleration: bool

  # the point around which to compute the acceleration, with respect to the
  # tray's origin
  com: list of float, length 3

  # instead of aligning with the acceleration vector, set to `true` to align
  # with `contact_plane_normal`, expressed in the world frame
  # this is useful for keeping the tray flat
  align_with_fixed_vector: bool

  # the world-frame normal vector to align with if `align_with_fixed_vector` is
  # true
  contact_plane_normal: list of float, length 3

# for safety, the end effector can be restricted to lie inside of a box
end_effector_box_constraint:
  enabled: bool                       # set `true` to enable this constraint in the controller
  xyz_lower: list of float, length 3  # task-space lower bound
  xyz_upper: list of float, length 3  # task-space upper bound

# settings for constraining the robot to avoid the path of a projectile
projectile_path_constraint:
  enabled: bool  # set `true` to enable this constraint in the controller

  # the names of the links to constrain to avoid collision with the projectile
  collision_links: list of str

  # the minimum distance that should be maintained between the projectile and
  # each link in `collision_links`
  distances: list of float

  # scale the constraints by this value
  scale: float, non-negative

# list of waypoints defining the trajectory relative to the EE's initial pose
waypoints:
  -
     # time the EE should be at this waypoint
     time: float, non-negative

     # EE position (x, y, z)
     position: list of float, length 3

     # EE orientation quaternion (x, y, z, w)
     orientation: list of float, length 4

# settings for general obstacle avoidance
obstacles
  enabled: bool         # set `true` to enable this constraint in the controller
  constraint_type: hard | soft  # `soft` not fully supported

  # minimum distance to enforce between objects
  minimum_distance: float, positive

  # list of pairs of str
  # defines the pairs of links which should be constrained not to collide
  collision_pairs:
    - [link1, link2]  # for example

  # list of dynamic obstacles
  dynamic:
    -
      name: str                # name of the dynamic obstacle
      radius: float, positive  # radius of the dynamic obstacle

      # modes of the obstacle
      modes:
        -
          # this mode is active from this time until mode with the next time
          time: float, non-negative

          # state of the obstacle at the start of the mode
          position: list of float, length 3
          velocity: list of float, length 3
          acceleration: list of float, length 3
    

# the controller can be initialized around some operating points
# may not fully supported
operating_points
  enabled: bool  # leave `false`

# list of the balanced object parameters from the perspective of the controller
objects:
  object_name: # name of the object

    # type of shape
    shape: cuboid | cylinder | wedge

    # parameters for different shape types
    side_lengths: list of float, length 3  # side lengths of a cuboid or wedge
    radius: float, non-negative            # radius of a cylinder
    height: float, non-negative            # height of a cylinder

    # offset of the center of mass from the shape's centroid
    com_offset: list of float, length 3

    # that mass of the object
    mass: float, non-negative

    # the diagonal of the inertia matrix about the CoM
    # this is optional; if not specified, shape is assumed to have uniform
    # density
    inertia_diag: list of float, length 3, non-negative

# parameters of the robot
robot:
  x0: list of float  # initial state

  # system dimensions
  dims:
    q: int, positive  # generalized position dimension
    v: int, positive  # generalized velocity dimension
    x: int, positive  # state dimension
    u: int, positive  # input dimension

  # URDF model defining the robot
  urdf:
    package: str  # package name
    path: str     # where to write the URDF relative to the package

    # the xacro (which is a superset of plain URDF) files to include together
    # to compile the URDF
    includes: list of str

    # specify any xacro argument values here
    args:
      arg: value

  # the name of the link representing the tool (i.e., the tray)
  tool_link_name: str

  # type of mobile base
  base_type: omnidirectional | fixed | nonholonomic | floating

  # optional map of joint names to the value at which they should remain constant
  # this is useful if one only wants to work with a subset of the whole robot
  locked_joints:
    joint_name: float

# weights on state, input, and end effector pose
weights
  input:  # input weight
    scale: float, non-negative         # scale coefficient for the whole weight matrix
    diag: list of float, non-negative  # weight matrix diagonal
  state:  # state weight
    scale: float, non-negative
    diag: list of float, non-negative
  end_effector:  # EE pose weight
    scale: float, non-negative
    diag: list of float, non-negative, length 3  # 3 position DOFs, 3 orientation DOFs

# state and input limits
limits:
  constraint_type: hard | soft
  input:  # input limits
    lower: list of float
    upper: list of float
  state:  # state limits
    lower: list of float
    upper: list of float

# settings for performing the forward rollout of the trajectory
# see <ocs2_oc/rollout/RolloutSettings.h>
rollout

arrangements
```

### Simulation
Nested under `simulation` are the following keys:
```yaml

timestep: float, non-negative     # simulation timestep [s]
duration: float, non-negative     # duration of the simuation [s]
gravity: list of float, length 3  # gravity vector

# name of the arrangement of objects being balanced
arrangement: str

# set `true` to show the contact points between objects in their initial
# position useful for debugging
show_contact_points: bool

# set `true` to show reference frames including the initial EE pose and all
# desired waypoints
# useful for debugging
show_debug_frame: bool

# define virtual cameras to capture static shots of the scene or as a viewpoint
# for a video
# cameras can be defined in multiple ways
cameras:
  # using absolute target and camera position
  camera_name:  # name of the camera
    target: list of float, length 3
    position: list of float, length 3

  # using target and camera position relative to EE initial position
  camera_name:
    relative_target: list of float, length 3
    relative_position: list of float, length 3

  # using target and distance and orientation of the camera
  # this one is convenient because these parameters can be read off of the
  # PyBullet GUI
  camera_name:
    target: list of float, length 3
    distance: float, non-negative
    roll: float
    pitch: float
    yaw: float

# define videos to be captured during the simulation
# this needs to be enabled by the command line argument --video
video:
  # set `true` to also save an image for each of the video frames (in addition
  # to the video itself)
  # this is useful for post-processing the video, or taking stills of
  # particular frames
  safe_frames: bool

  # absolute path to the directory in which to save the video
  dir: str

  # take a new frame every `timestep` seconds
  timestep: float, non-negative [s]

  # multiple videos (i.e., from different viewpoints) can be taken simultaneously
  # each one is specifed here
  views:
    -
      camera: str  # the name of the camera to use to record the video
      name: str    # the name of the video corresponding to this view

# define photos to be taken during the simulation
# currently only photos at the start or end of the simulation are supported
photos:
  start:  # viewpoints to take photos at the start
    -
      camera: str
      name: str
  end:  # viewpoints to take photos at the end
    -
      camera: str
      name: str

static_obstacles:
  enabled: bool  # add static obstacles to the simulation

  # obstacles are defined using a URDF
  urdf:
    package: str  # name of the package
    path: str     # where to write the URDF relative to the package

    # the xacro (which is a superset of plain URDF) files to include together
    # to compile the URDF
    includes: list of str

dynamic_obstacles:
  enabled: bool  # add dynamic obstacles to the simulation

  # list of dynamic obstacles
  obstacles:
    -
      # set `true` if the obstacle's trajectory should be actively be
      # controlled/tracked, or `false` if it should be left subject to the
      # simulation's dynamics
      controlled: bool

      # radius of the obstacle
      radius: float, positive

      # `position` of each mode is relative to the initial EE if `true`,
      # otherwise relative to world coordinates
      relative: bool

      # same as in the `controller.obstacles.dynamic` section
      modes:

# parameters of the robot
# this is the same as in the `controller` section with the following exceptions:
robot:
  ...
  # instead of `x0`, the simulation only requires the home joint configuration
  home: list of floats

  # measurement and process noise
  noise:
    measurement:
      q_std_dev: float, non-negative  # standard deviation of measured joint positions
      v_std_dev: float, non-negative  # standard deviation of measured joint velocities
    process:
      v_std_dev: float, non-negative  # standard deviation of velocity inputs

  # list of the names of the joints being controlled
  joint_names: list of str

# the objects are defined in the same way as in the `controller` section,
# except that there is color parameter as well:
objects
  object_name:
    ...
    # [r, g, b, a] color, where each value is between 0 and 1, a is the alpha
    # (transparency)
    color: list of float

# this has the same structure as in the `controller` section
# we typically make a seperate arrangements.yaml file that is included in both
# the controller and simulation settings, where the parameters of the objects
# themselves are changed if we want to have differences between the two
arrangements
```
