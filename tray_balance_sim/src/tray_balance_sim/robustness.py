import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pybullet as pyb
import liegroups
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree

from tray_balance_sim import util, clustering
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

    # reset velocity to zero so we're in a resting state when we start the
    # trajectory proper
    robot.command_velocity(np.zeros(robot.ni))
    pyb.stepSimulation()


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


def get_T_we(robot):
    r_ew_w, Q_we = robot.link_pose()
    return liegroups.SE3(
        rot=liegroups.SO3.from_quaternion(Q_we, ordering="xyzw"), trans=r_ew_w
    )


def set_bounding_spheres(
    robot,
    objects,
    settings,
    target,
    sim_timestep,
    k=2,
    num_images=4,
    plot_point_cloud=False,
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

    # rotate EE around z-axis for a total rotation of `total_angle`, taking
    # `num_images` depth images
    total_angle = -2 * np.pi
    angle_increment = total_angle / num_images
    points_list = []
    for i in range(num_images):
        # get points (in world frame) and transform them into EE frame
        points_w = get_object_point_cloud(camera, objects)
        points_e = get_T_we(robot).inv().dot(points_w)
        points_list.append(points_e)

        # we do an extra rotation at the end, but that's fine
        rotate_end_effector(
            robot,
            angle=angle_increment,
            sim_timestep=sim_timestep,
            duration=3.0,
            realtime=False,
        )

    points = np.vstack(points_list)

    # rotate back to initial position
    rotate_end_effector(
        robot,
        angle=-total_angle,
        sim_timestep=sim_timestep,
        duration=12.0,
        realtime=False,
    )

    # compute max_radius for robust inertia
    max_radius = 0.5 * np.max(pdist(points))
    print(f"max_radius = {max_radius}")

    # remove points within a given radius of other points
    tree = KDTree(points)
    remove = np.zeros(points.shape[0], dtype=bool)
    for i in range(points.shape[0]):
        if not remove[i]:
            idx = tree.query_ball_point(points[i, :], r=0.005)
            for j in idx:
                if j != i:
                    remove[j] = True
    points = points[~remove, :]

    # cluster point cloud points and bound with spheres
    centers, radii, assignments = clustering.cluster_and_bound(
        points, k=k, cluster_type="greedy", bound_type="fischer", n=1
    )

    # also rotate camera target into EE frame
    target_e = get_T_we(robot).inv().dot(camera.target)

    if plot_point_cloud:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for i in range(k):
            (idx,) = np.nonzero(assignments == i)
            ax.scatter(points[idx, 0], points[idx, 1], zs=points[idx, 2])
        ax.scatter(target_e[0], target_e[1], zs=target_e[2])
        # for i in range(k):
        #     ax.scatter(
        #         cluster_centers[i, 0],
        #         cluster_centers[i, 1],
        #         zs=cluster_centers[i, 2],
        #         color="k",
        #     )
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
        collision_sphere = ocs2.CollisionSphere(
            name="robust_collision_sphere_" + str(i),
            parent_frame_name="thing_tool",
            offset=centers[i, :],
            radius=radii[i],
        )
        settings.dynamic_obstacle_settings.collision_spheres.push_back(collision_sphere)
        settings.collision_avoidance_settings.extra_spheres.push_back(collision_sphere)

    settings.tray_balance_settings.robust_params.balls = balls
    settings.tray_balance_settings.robust_params.max_radius = max_radius
