from upright_sim import simulation
import numpy as np
import pybullet as pyb
import fcl
import scipy
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from functools import partial

import IPython

MASS = 1.0
MU = 1.0


def box(side_lengths, position, color):
    side_lengths = np.array(side_lengths)
    position = np.array(position)

    # box = simulation.BulletBody.cuboid(MASS, MU, side_lengths, color=color)
    # box.add_to_sim(position)
    return fcl.CollisionObject(fcl.Box(*side_lengths), fcl.Transform(position))


def cylinder(radius, height, position, color):
    position = np.array(position)

    # cy = simulation.BulletBody.cylinder(MASS, MU, radius, height, color=color)
    # cy.add_to_sim(position)
    return fcl.CollisionObject(fcl.Cylinder(radius, height), fcl.Transform(position))


def main():
    np.set_printoptions(precision=8, suppress=True)

    # pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    # pyb.resetDebugVisualizerCamera(
    #     cameraDistance=4,
    #     cameraYaw=42,
    #     cameraPitch=-35.8,
    #     cameraTargetPosition=[1.28, 0.045, 0.647],
    # )
    # pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)

    o1 = box([0.2, 0.2, 0.2], [0, 0, 0.1], color=(1, 0, 0, 0.5))
    o2 = box([0.2, 0.2, 0.2], [0.3, 0.1, 0.1], color=(0, 0, 1, 0.5))
    o3 = box([0.4, 0.2, 0.1], [0.2, 0, 0.25], color=(0, 1, 0, 0.5))

    req = fcl.CollisionRequest()
    res = fcl.CollisionResult()
    ret = fcl.collide(o1, o3, req, res)

    # colors = [[0, 0, 0] for _ in vs]
    # pyb.addUserDebugPoints(vs, colors, pointSize=10)

    IPython.embed()


if __name__ == "__main__":
    main()
