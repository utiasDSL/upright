import numpy as np
import pinocchio
import hppfcl as fcl

import upright_core as core
from upright_control import bindings
from mobile_manipulation_central.kinematics import RobotKinematics


def _build_pinocchio_model(urdf_path, base_type, locked_joints, base_pose):
    """Build Pinocchio robot model."""
    root_joint = pinocchio.JointModelComposite(3)
    root_joint.addJoint(pinocchio.JointModelPX())
    root_joint.addJoint(pinocchio.JointModelPY())
    root_joint.addJoint(pinocchio.JointModelRZ())
    model = pinocchio.buildModelFromUrdf(urdf_path, root_joint)

    # lock the base joint after setting the pose
    if base_type == bindings.RobotBaseType.Fixed:
        joint_idx = model.getJointId("root_joint")
        joint_ids_to_lock = [joint_idx]
        q_idx = model.idx_qs[joint_idx]

        q = np.zeros(model.nq)
        q[q_idx : q_idx + 3] = base_pose

        model = pinocchio.buildReducedModel(model, joint_ids_to_lock, q)

    # lock joints if needed
    if locked_joints is not None and len(locked_joints) > 0:
        # it doesn't matter what value for set for the free joints here; this
        # is updated later
        q = np.zeros(model.nq)
        joint_ids_to_lock = []
        for name, value in locked_joints.items():
            joint_idx = model.getJointId(name)
            joint_ids_to_lock.append(joint_idx)
            q_idx = model.idx_qs[joint_idx]
            q[q_idx] = value
        model = pinocchio.buildReducedModel(model, joint_ids_to_lock, q)

    return model


def _build_dynamic_obstacle_model(obstacles):
    """Build model of dynamic obstacles."""
    model = pinocchio.Model()
    model.name = "dynamic_obstacles"
    geom_model = pinocchio.GeometryModel()

    for obstacle in obstacles:
        # free-floating joint
        joint_name = obstacle.name + "_joint"
        joint_placement = pinocchio.SE3.Identity()
        joint_id = model.addJoint(
            0, pinocchio.JointModelTranslation(), joint_placement, joint_name
        )

        # body
        mass = 1.0
        inertia = pinocchio.Inertia.FromSphere(mass, obstacle.radius)
        body_placement = pinocchio.SE3.Identity()
        model.appendBodyToJoint(joint_id, inertia, body_placement)

        # visual model
        shape = fcl.Sphere(obstacle.radius)
        geom_obj = pinocchio.GeometryObject(
            obstacle.name, joint_id, shape, body_placement
        )
        geom_obj.meshColor = np.ones((4))
        geom_model.addGeometryObject(geom_obj)

    return model, geom_model


def _append_model(
    robot, geom, model, geom_model, frame_index=0, placement=pinocchio.SE3.Identity()
):
    """Append models to create new models."""
    new_model, new_collision_model = pinocchio.appendModel(
        robot.model, model, geom.collision_model, geom_model, frame_index, placement
    )
    _, new_visual_model = pinocchio.appendModel(
        robot.model, model, geom.visual_model, geom_model, frame_index, placement
    )

    new_robot = UprightRobotKinematics(new_model, robot.mapping, robot.tool_link_name)
    new_geom = UprightRobotGeometry(new_robot, new_collision_model, new_visual_model)
    return new_robot, new_geom


def build_robot_interfaces(settings):
    """Build robot and geometry interface from control settings."""
    # build robot
    model = _build_pinocchio_model(
        settings.robot_urdf_path,
        base_type=settings.robot_base_type,
        locked_joints=settings.locked_joints,
        base_pose=settings.base_pose,
    )
    mapping = bindings.SystemPinocchioMapping(settings.dims)
    robot = UprightRobotKinematics(
        model=model,
        mapping=mapping,
        tool_link_name=settings.end_effector_link_name,
    )

    # build geometry
    geom = UprightRobotGeometry.from_robot_and_urdf(robot, settings.robot_urdf_path)

    # add a ground plane
    ground_placement = pinocchio.SE3.Identity()
    ground_shape = fcl.Halfspace(np.array([0, 0, 1]), 0)
    ground_geom_obj = pinocchio.GeometryObject(
        "ground", model.frames[0].parent, ground_shape, ground_placement
    )
    ground_geom_obj.meshColor = np.ones((4))

    # we don't add as a visual object because it is not supported by the
    # meshcat viewer
    geom.add_collision_objects([ground_geom_obj])

    # add dynamic obstacles
    if len(settings.obstacle_settings.dynamic_obstacles) > 0:
        obs_model, obs_geom_model = _build_dynamic_obstacle_model(
            settings.obstacle_settings.dynamic_obstacles
        )
        robot, geom = _append_model(robot, geom, obs_model, obs_geom_model)

    # add static obstacles
    if len(settings.obstacle_settings.obstacle_urdf_path) > 0:
        geom.add_geometry_objects_from_urdf(
            settings.obstacle_settings.obstacle_urdf_path
        )

    # add link pairs to check for collision
    pairs = [pair for pair in settings.obstacle_settings.collision_link_pairs]
    geom.add_collision_pairs(pairs)

    return robot, geom


class UprightRobotGeometry:
    """Visual and collision geometry models for a robot and environment."""

    def __init__(self, robot, collision_model, visual_model):
        self.robot = robot
        self.collision_model = collision_model
        self.visual_model = visual_model

        # get rid of automatically-added collision pairs
        self.collision_model.removeAllCollisionPairs()

    @classmethod
    def from_robot_and_urdf(cls, robot, urdf_path):
        collision_model = pinocchio.GeometryModel()
        visual_model = pinocchio.GeometryModel()
        geom = cls(robot, collision_model, visual_model)
        geom.add_geometry_objects_from_urdf(urdf_path, model=robot.model)
        return geom

    def add_collision_objects(self, geoms):
        """Add a list of geometry objects to the model."""
        for geom in geoms:
            self.collision_model.addGeometryObject(geom)

    def add_visual_objects(self, geoms):
        """Add a list of geometry objects to the model."""
        for geom in geoms:
            self.visual_model.addGeometryObject(geom)

    def add_geometry_objects_from_urdf(self, urdf_path, model=None):
        """Add geometry objects from a URDF file."""
        geom_model = pinocchio.GeometryModel()

        if model is None:
            model = pinocchio.buildModelFromUrdf(urdf_path)

        # load collision objects
        pinocchio.buildGeomFromUrdf(model, urdf_path, pinocchio.COLLISION, geom_model)
        self.add_collision_objects(geom_model.geometryObjects)

        # load visual objects
        pinocchio.buildGeomFromUrdf(model, urdf_path, pinocchio.VISUAL, geom_model)
        self.add_visual_objects(geom_model.geometryObjects)

    def add_collision_pairs(self, pairs):
        """Add collision pairs to the model."""
        for pair in pairs:
            id1 = self.collision_model.getGeometryId(pair[0])
            id2 = self.collision_model.getGeometryId(pair[1])
            self.collision_model.addCollisionPair(pinocchio.CollisionPair(id1, id2))

    def compute_distances(self):
        """Compute distances between collision pairs.

        robot.forward(...) should be called first.
        """
        geom_data = pinocchio.GeometryData(self.collision_model)

        pinocchio.updateGeometryPlacements(
            self.robot.model, self.robot.data, self.collision_model, geom_data
        )
        pinocchio.computeDistances(self.collision_model, geom_data)

        return np.array([result.min_distance for result in geom_data.distanceResults])

    def visualize(self, q):
        """Visualize the robot using meshcat."""
        viz = pinocchio.visualize.MeshcatVisualizer(
            self.robot.model, self.collision_model, self.visual_model
        )
        viz.initViewer()
        viz.loadViewerModel()
        viz.display(q)
        return viz


class UprightRobotKinematics(RobotKinematics):
    def __init__(self, model, mapping, tool_link_name):
        super().__init__(model, tool_link_name)
        self.mapping = mapping

    def forward_xu(self, x, u=None):
        """Forward kinematics. Must be called before the link pose, velocity,
        or acceleration methods."""
        if u is None:
            u = np.zeros(self.nv)

        # get values in Pinocchio coordinates
        q = self.mapping.get_pinocchio_joint_position(x)
        v = self.mapping.get_pinocchio_joint_velocity(x, u)
        a = self.mapping.get_pinocchio_joint_acceleration(x, u)

        self.forward(q, v, a)

    def forward_derivatives_xu(self, x, u=None):
        # get values in Pinocchio coordinates
        q = self.mapping.get_pinocchio_joint_position(x)
        v = self.mapping.get_pinocchio_joint_velocity(x, u)
        a = self.mapping.get_pinocchio_joint_acceleration(x, u)

        self.forward_derivatives(q, v, a)
