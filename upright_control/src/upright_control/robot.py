import numpy as np
import pinocchio

import upright_core as core
from upright_control import bindings


class PinocchioGeometry:
    def __init__(self, robot):
        self.robot = robot
        self.collision_model = pinocchio.GeometryModel()
        self.visual_model = pinocchio.GeometryModel()

        self.add_geometry_objects_from_urdf(robot.urdf_path, model=robot.model)

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

    def add_geometry_objects_from_config(self, config):
        """Add geometry objects from a config dict.

        Expects the config to have a key "urdf" with the usual sub-dict.
        """
        urdf_path = core.parsing.parse_ros_path(config["urdf"])
        self.add_geometry_objects_from_urdf(urdf_path)

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


class PinocchioRobot:
    def __init__(self, config):
        # dimensions
        self.dims = bindings.RobotDimensions()
        self.dims.q = config["dims"]["q"]  # num positions
        self.dims.v = config["dims"]["v"]  # num velocities
        self.dims.x = config["dims"]["x"]  # num states
        self.dims.u = config["dims"]["u"]  # num inputs

        # create the model
        self.urdf_path = core.parsing.parse_ros_path(config["urdf"])
        base_type = config["base_type"]
        if base_type == "fixed":
            self.model = pinocchio.buildModelFromUrdf(self.urdf_path)
            self.mapping = bindings.FixedBasePinocchioMapping(self.dims)
        elif base_type == "omnidirectional":
            root_joint = pinocchio.JointModelComposite(3)
            root_joint.addJoint(pinocchio.JointModelPX())
            root_joint.addJoint(pinocchio.JointModelPY())
            root_joint.addJoint(pinocchio.JointModelRZ())
            # root_joint.addJoint(pinocchio.JointModelRUBZ())
            # root_joint = pinocchio.JointModelPlanar()
            self.model = pinocchio.buildModelFromUrdf(self.urdf_path, root_joint)
            self.mapping = bindings.OmnidirectionalPinocchioMapping(self.dims)
        else:
            raise ValueError(f"Invalid base type {base_type}.")

        self.data = self.model.createData()

        self.tool_idx = self.model.getBodyId(config["tool_link_name"])

    def forward_qva(self, q, v=None, a=None):
        """Forward kinematics using (q, v, a) all in the world frame (i.e.,
        corresponding directly to the Pinocchio model."""
        if v is None:
            v = np.zeros(self.dims.v)
        if a is None:
            a = np.zeros(self.dims.v)

        assert q.shape == (self.dims.q,)
        assert v.shape == (self.dims.v,)
        assert a.shape == (self.dims.v,)

        pinocchio.forwardKinematics(self.model, self.data, q, v, a)
        pinocchio.updateFramePlacements(self.model, self.data)

    def forward_derivatives_qva(self, q, v, a=None):
        """Compute derivatives of the forward kinematics using (q, v, a) all in
        the world frame (i.e., corresponding directly to the Pinocchio
        model."""
        if a is None:
            a = np.zeros(self.dims.v)

        assert q.shape == (self.dims.q,)
        assert v.shape == (self.dims.v,)
        assert a.shape == (self.dims.v,)

        pinocchio.computeForwardKinematicsDerivatives(self.model, self.data, q, v, a)

    def forward(self, x, u=None):
        """Forward kinematics. Must be called before the link pose, velocity,
        or acceleration methods."""
        if u is None:
            u = np.zeros(self.dims.u)

        assert x.shape == (self.dims.x,)
        assert u.shape == (self.dims.u,)

        # get values in Pinocchio coordinates
        q = self.mapping.get_pinocchio_joint_position(x)
        v = self.mapping.get_pinocchio_joint_velocity(x, u)
        a = self.mapping.get_pinocchio_joint_acceleration(x, u)

        self.forward_qva(q, v, a)

    def forward_derivatives(self, x, u=None):
        if u is None:
            u = np.zeros(self.dims.u)

        assert x.shape == (self.dims.x,)
        assert u.shape == (self.dims.u,)

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
        dAs_dq = dAdq + np.vstack((dcdq, np.zeros((3, self.dims.q))))
        dAs_dv = dAdv + np.vstack((dcdv, np.zeros((3, self.dims.v))))
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

    # TODO need to update these methods from previous iteration of the model
    # def tangent(self, x, u):
    #     """Tangent vector dx = f(x, u)."""
    #     B = block_diag(rot2d(x[2], np=jnp), jnp.eye(7))
    #     return jnp.concatenate((x[self.ni :], B @ u))
    #
    # def simulate(self, x, u):
    #     """Forward simulate the model."""
    #     # TODO not sure if I can somehow use RK4 for part and not for
    #     # all---we'd have to split base and arm
    #     k1 = self.tangent(x, u)
    #     k2 = self.tangent(x + self.dt * k1 / 2, u)
    #     k3 = self.tangent(x + self.dt * k2 / 2, u)
    #     k4 = self.tangent(x + self.dt * k3, u)
    #     return x + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
