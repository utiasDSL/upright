import argparse

import pybullet as pyb
import numpy as np
from scipy.linalg import expm
import yaml

import upright_core as core

import IPython


def right_triangular_prism_mesh(half_extents):
    vertices, _ = core.right_triangle.right_triangular_prism_vertices_normals(half_extents)

    # fmt: off
    indices = np.array([
        [0, 1, 2],
        [0, 1, 4],
        [0, 4, 3],
        [0, 3, 5],
        [0, 5, 2],
        [1, 4, 5],
        [1, 5, 2],
        [3, 4, 5]])
    # fmt: on

    # duplicate vertices with opposite winding, so the object is visible from
    # both sides
    indices = np.vstack((indices, np.flip(indices, axis=1))).flatten()
    return vertices, list(indices)


def right_triangular_prism(mass, half_extents, position=None, orientation=None):
    if position is None:
        position = [0, 0, 0]
    if orientation is None:
        orientation = [0, 0, 0, 1]

    vertices, indices = right_triangular_prism_mesh(half_extents)

    hx, hy, hz = half_extents
    D, C = core.right_triangle.right_triangular_prism_inertia_normalized(half_extents)

    inertial_position = [-hx / 3, 0, -hz / 3]
    local_inertia_diagonal = mass * np.diag(D)
    inertial_orientation = core.math.rot_to_quat(C)

    col_id = pyb.createCollisionShape(pyb.GEOM_MESH, vertices=vertices, indices=indices)
    vis_id = pyb.createVisualShape(
        pyb.GEOM_MESH, vertices=vertices, indices=indices, rgbaColor=(1, 0, 0, 1)
    )

    body_id = pyb.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        baseInertialFramePosition=inertial_position,
        baseInertialFrameOrientation=inertial_orientation,
        basePosition=position,
        baseOrientation=orientation,
    )
    pyb.changeDynamics(body_id, -1, localInertiaDiagonal=local_inertia_diagonal)
    return body_id


# TODO: unused
def dhtf(q, a, d, α):
    """Constuct a transformation matrix from D-H parameters."""
    cα = np.cos(α)
    sα = np.sin(α)
    cq = np.cos(q)
    sq = np.sin(q)
    return np.array(
        [
            [cq, -sq * cα, sq * sα, a * cq],
            [sq, cq * cα, -cq * sα, a * sq],
            [0, sα, cα, d],
            [0, 0, 0, 1],
        ]
    )


def zoh(A, B, dt):
    """Compute discretized system matrices assuming zero-order hold on input."""
    ra, ca = A.shape
    rb, cb = B.shape

    assert ra == ca  # A is square
    assert ra == rb  # B has same number of rows as A

    ch = ca + cb
    rh = ch

    H = np.block([[A, B], [np.zeros((rh - ra, ch))]])
    Hd = expm(dt * H)
    Ad = Hd[:ra, :ca]
    Bd = Hd[:rb, ca : ca + cb]

    return Ad, Bd


def draw_curve(waypoints, rgb=(1, 0, 0), dist=0.05, linewidth=1, dashed=False):
    """Draw debug lines along a curve represented by waypoints in PyBullet."""
    # process waypoints to space them (roughly) evenly
    visual_points = [waypoints[0, :]]
    for i in range(1, len(waypoints)):
        d = np.linalg.norm(waypoints[i, :] - visual_points[-1])
        if d >= dist:
            visual_points.append(waypoints[i, :])

    step = 2 if dashed else 1
    for i in range(0, len(visual_points) - 1, step):
        start = visual_points[i]
        end = visual_points[i + 1]
        pyb.addUserDebugLine(
            list(start),
            list(end),
            lineColorRGB=rgb,
            lineWidth=linewidth,
        )
