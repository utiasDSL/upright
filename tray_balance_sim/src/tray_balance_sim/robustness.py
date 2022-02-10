import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pybullet as pyb
import liegroups
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree

from tray_balance_sim import util, pyb_util, clustering
from tray_balance_sim.camera import Camera

import tray_balance_ocs2.MobileManipulatorPythonInterface as ocs2

import IPython


def rotate_end_effector(robot, angle, duration, sim_timestep, realtime=True):
    u = np.zeros(robot.ni)
    u[-1] = angle / duration
    robot.command_velocity(u)

    t = 0
    while t < duration:
        if realtime:
            time.sleep(sim_timestep)
        pyb.stepSimulation()
        t += sim_timestep


def get_object_point_cloud(camera, objects):
    w, h, rgb, dep, seg = camera.get_frame()
    # camera.save_frame("testframe.png")
    points = camera.get_point_cloud(dep)

    # mask out everything except balanced objects
    mask = np.zeros_like(seg)
    for obj in objects.values():
        mask = np.logical_or(seg == obj.bullet.uid, mask)
    points = points[mask.T, :]

    return points


def set_bounding_spheres(
    robot, objects, settings, target, sim_timestep, k=2, plot_point_cloud=False
):
    cam_pos = [target[0], target[1] - 1, target[2]]
    camera = Camera(
        camera_position=cam_pos,
        target_position=target,
        width=200,
        height=200,
        fov=50,
        near=0.1,
        far=5,
    )
    points1_w = get_object_point_cloud(camera, objects)

    # TODO abstract this away
    r_ew_w, Q_we = robot.link_pose()
    T_ew1 = liegroups.SE3(
        rot=liegroups.SO3.from_quaternion(Q_we, ordering="xyzw"), trans=r_ew_w
    ).inv()
    points1_e = T_ew1.dot(points1_w)

    # rotate to take picture of the other side of the objects
    rotate_end_effector(
        robot, angle=-np.pi, sim_timestep=sim_timestep, duration=5.0, realtime=False
    )

    points2_w = get_object_point_cloud(camera, objects)

    r_ew_w, Q_we = robot.link_pose()
    T_ew2 = liegroups.SE3(
        rot=liegroups.SO3.from_quaternion(Q_we, ordering="xyzw"), trans=r_ew_w
    ).inv()
    points2_e = T_ew2.dot(points2_w)

    points_e = np.vstack((points1_e, points2_e))

    # rotate back to initial position
    rotate_end_effector(
        robot, angle=np.pi, sim_timestep=sim_timestep, duration=5.0, realtime=False
    )

    # compute max_radius for robust inertia
    max_radius = 0.5 * np.max(pdist(points_e))
    print(f"max_radius = {max_radius}")

    # remove points within a given radius of other points
    tree = KDTree(points_e)
    remove = np.zeros(points_e.shape[0], dtype=bool)
    for i in range(points_e.shape[0]):
        if not remove[i]:
            idx = tree.query_ball_point(points_e[i, :], r=0.005)
            for j in idx:
                if j != i:
                    remove[j] = True
    points_e = points_e[~remove, :]

    # cluster point cloud points and bound with spheres
    centers, radii = clustering.cluster_and_bound(
        points_e, k=k, cluster_type="greedy", bound_type="fischer"
    )

    # also rotate camera target into EE frame
    target_e = T_ew1.dot(camera.target)

    if plot_point_cloud:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(points_e[:, 0], points_e[:, 1], zs=points_e[:, 2])
        ax.scatter(target_e[0], target_e[1], zs=target_e[2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    r_ew_w, Q_we = robot.link_pose()
    C_we = util.quaternion_to_matrix(Q_we)

    # balls are already in the EE frame, as required
    balls = []
    for i in range(k):
        balls.append(ocs2.Ball(centers[i, :], radii[i]))
    settings.tray_balance_settings.robust_params.balls = balls
    settings.tray_balance_settings.robust_params.max_radius = max_radius


# TODO: could build a generic object to attach a visual object to a multibody
class RobustSpheres:
    def __init__(self, robot, robust_params, color=(0.5, 0.5, 0.5, 0.5)):
        self.robot = robot
        r_ew_w, _ = robot.link_pose()

        self.spheres = []
        self.centers = []
        for ball in robust_params.balls:
            position = r_ew_w + ball.center
            self.centers.append(ball.center)
            self.spheres.append(
                pyb_util.GhostSphere(radius=ball.radius, position=position, color=color)
            )

    def update(self):
        r_ew_w, Q_we = self.robot.link_pose()
        for i in range(len(self.spheres)):
            position = util.transform_point(r_ew_w, Q_we, self.centers[i])
            self.spheres[i].set_position(position)
