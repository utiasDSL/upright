import numpy as np
import pinocchio
import hppfcl as fcl

import upright_core as core
from upright_control import bindings

import IPython


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
        robot.model, model, geom.collision_model, geom_model, 0, placement
    )
    _, new_visual_model = pinocchio.appendModel(
        robot.model, model, geom.visual_model, geom_model, 0, placement
    )

    new_robot = PinocchioRobot(new_model, robot.mapping, robot.tool_link_name)
    new_geom = PinocchioGeometry(new_robot, new_collision_model, new_visual_model)
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
    robot = PinocchioRobot(
        model=model,
        mapping=mapping,
        tool_link_name=settings.end_effector_link_name,
    )

    # build geometry
    geom = PinocchioGeometry.from_robot_and_urdf(robot, settings.robot_urdf_path)

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


class PinocchioGeometry:
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


# TODO revise to inherit from RobotKinematics in mm_central
class PinocchioRobot:
    def __init__(self, model, mapping, tool_link_name):
        self.model = model
        self.mapping = mapping
        self.data = self.model.createData()
        self.tool_link_name = tool_link_name
        self.tool_idx = self.model.getBodyId(tool_link_name)
        self.nq = self.model.nq
        self.nv = self.model.nv

    def forward_qva(self, q, v=None, a=None):
        """Forward kinematics using (q, v, a) all in the world frame (i.e.,
        corresponding directly to the Pinocchio model."""
        if v is None:
            v = np.zeros(self.nv)
        if a is None:
            a = np.zeros(self.nv)

        assert q.shape == (self.nq,)
        assert v.shape == (self.nv,)
        assert a.shape == (self.nv,)

        pinocchio.forwardKinematics(self.model, self.data, q, v, a)
        pinocchio.updateFramePlacements(self.model, self.data)

    def forward_derivatives_qva(self, q, v, a=None):
        """Compute derivatives of the forward kinematics using (q, v, a) all in
        the world frame (i.e., corresponding directly to the Pinocchio
        model."""
        if a is None:
            a = np.zeros(self.nv)

        assert q.shape == (self.nq,)
        assert v.shape == (self.nv,)
        assert a.shape == (self.nv,)

        pinocchio.computeForwardKinematicsDerivatives(self.model, self.data, q, v, a)

    def forward(self, x, u=None):
        """Forward kinematics. Must be called before the link pose, velocity,
        or acceleration methods."""
        if u is None:
            u = np.zeros(self.nv)

        # get values in Pinocchio coordinates
        q = self.mapping.get_pinocchio_joint_position(x)
        v = self.mapping.get_pinocchio_joint_velocity(x, u)
        a = self.mapping.get_pinocchio_joint_acceleration(x, u)

        self.forward_qva(q, v, a)

    def forward_derivatives(self, x, u=None):
        # get values in Pinocchio coordinates
        q = self.mapping.get_pinocchio_joint_position(x)
        v = self.mapping.get_pinocchio_joint_velocity(x, u)
        a = self.mapping.get_pinocchio_joint_acceleration(x, u)

        self.forward_derivatives_qva(q, v, a)

    def link_pose(self, link_idx=None):
        if link_idx is None:
            link_idx = self.tool_idx
        pose = self.data.oMf[link_idx]
        r = pose.translation
        Q = core.math.rot_to_quat(pose.rotation)
        return r.copy(), Q.copy()

    def link_velocity(self, link_idx=None):
        if link_idx is None:
            link_idx = self.tool_idx
        V = pinocchio.getFrameVelocity(
            self.model,
            self.data,
            link_idx,
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return V.linear, V.angular

    def link_acceleration(self, link_idx=None):
        if link_idx is None:
            link_idx = self.tool_idx
        A = pinocchio.getFrameClassicalAcceleration(
            self.model,
            self.data,
            link_idx,
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return A.linear, A.angular

    def link_spatial_acceleration(self, link_idx=None):
        if link_idx is None:
            link_idx = self.tool_idx
        A = pinocchio.getFrameAcceleration(
            self.model,
            self.data,
            link_idx,
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return A.linear, A.angular

    def jacobian(self, q):
        return pinocchio.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.tool_idx,
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

    def link_velocity_derivatives(self, link_idx=None):
        """Compute derivative of link velocity with respect to q and v."""
        if link_idx is None:
            link_idx = self.tool_idx
        dVdq, dVdv = pinocchio.getFrameVelocityDerivatives(
            self.model,
            self.data,
            link_idx,
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return dVdq, dVdv

    def link_acceleration_derivatives(self, link_idx=None):
        """Compute derivative of link classical acceleration with respect to q, v, a."""
        dr, ω = self.link_velocity(link_idx=link_idx)
        dVdq, dVdv = self.link_velocity_derivatives(link_idx=link_idx)
        dAdq, dAdv, dAda = self.link_spatial_acceleration_derivatives(link_idx=link_idx)

        # derivative of the coriolis term
        ddrdq, dwdq = dVdq[:3, :], dVdq[3:, :]
        ddrdv, dwdv = dVdv[:3, :], dVdv[3:, :]
        dcdq = (np.cross(dwdq.T, dr) + np.cross(ω, ddrdq.T)).T
        dcdv = (np.cross(dwdv.T, dr) + np.cross(ω, ddrdv.T)).T

        # add the coriolis term to the spatial acceleration
        dAs_dq = dAdq + np.vstack((dcdq, np.zeros((3, self.nq))))
        dAs_dv = dAdv + np.vstack((dcdv, np.zeros((3, self.nv))))
        dAs_da = dAda

        return dAs_dq, dAs_dv, dAs_da

    def link_spatial_acceleration_derivatives(self, link_idx=None):
        """Compute derivative of link spatial acceleration with respect to q, v, a."""
        if link_idx is None:
            link_idx = self.tool_idx
        _, dAdq, dAdv, dAda = pinocchio.getFrameAccelerationDerivatives(
            self.model,
            self.data,
            link_idx,
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return dAdq, dAdv, dAda
