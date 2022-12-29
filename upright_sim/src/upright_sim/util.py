import pybullet as pyb
import numpy as np
import upright_core as core


def wedge_mesh(half_extents):
    """Generate vertices and indices required to build a mesh object in PyBullet."""
    wedge = core.polyhedron.ConvexPolyhedron.wedge(half_extents)

    # convert to raw list of lists
    vertices = [list(v) for v in wedge.vertices]

    # sets of vertices making up triangular faces
    # counter-clockwise winding about normal facing out of the shape
    # fmt: off
    indices = np.array([
        [0, 1, 2],
        [0, 4, 1],
        [0, 3, 4],
        [0, 5, 3],
        [0, 2, 5],
        [1, 4, 5],
        [1, 5, 2],
        [3, 5, 4]])
    # fmt: on

    return vertices, list(indices.flatten())


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
