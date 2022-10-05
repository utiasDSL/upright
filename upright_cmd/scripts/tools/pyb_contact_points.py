from upright_sim import simulation
import numpy as np
import pybullet as pyb
import hppfcl as fcl
import scipy
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from functools import partial

import IPython

MASS = 1.0
MU = 1.0


class Box:
    def __init__(self, side_lengths):
        self.side_lengths = np.array(side_lengths)

    def vertices(self, transform):
        a, b, c = 0.5 * self.side_lengths

        # vertices in the object frame
        # fmt: off
        vs = np.array([[a, b, c],
                       [a, b, -c],
                       [a, -b, c],
                       [a, -b, -c],
                       [-a, b, c],
                       [-a, b, -c],
                       [-a, -b, c],
                       [-a, -b, -c]])
        # fmt: on

        # apply the transform
        for i in range(vs.shape[0]):
            vs[i, :] = transform.transform(vs[i, :])
        return vs

    def collision_object(self, transform):
        return fcl.CollisionObject(fcl.Box(*self.side_lengths), transform)

    def constraints(self, x, c):
        # inside if all constraints >= 0
        up = c + 0.5 * self.side_lengths
        low = c - 0.5 * self.side_lengths
        return np.concatenate((up - x, x - low))

    def constraints_jac(self, x, c):
        return np.vstack((-np.eye(3), np.eye(3)))


class Cylinder:
    def __init__(self, radius, height):
        self.radius = radius
        self.height = height

        s = 0.5 * np.sqrt(2) * radius
        self.inner_box = Box([s, s, height])

    def vertices(self, transform):
        return self.inner_box.vertices(transform)

    def collision_object(self, transform):
        return fcl.CollisionObject(fcl.Cylinder(self.radius, self.height), transform)


def point_object_distance(point, obj):
    req = fcl.DistanceRequest()
    res = fcl.DistanceResult()
    p = fcl.CollisionObject(fcl.Sphere(0), fcl.Transform3f(point))
    dist = fcl.distance(p, obj, req, res)
    nearest_point = res.getNearestPoint2()
    return dist, nearest_point


def low_dim_convex_hull(points):
    c = np.mean(points, axis=0)
    _, d, VT = scipy.linalg.svd(points - c)
    r = np.sum(d > 1e-5)
    V = VT.T
    X = (points @ V)[:, :r]

    # only take the points that are at the vertices of the constraint manifold
    hull = ConvexHull(X)
    return points[hull.vertices]


def nearest_distance(o1, o2):
    req = fcl.DistanceRequest()
    res = fcl.DistanceResult()
    dist = fcl.distance(o1, o2, req, res)
    return dist, res.getNearestPoint2(), res.normal


def solve_vertex(o1, o2, c, x0):
    def cost(x):
        return c @ x

    def jac(x):
        return c

    def ineq_con(x):
        d1 = point_object_distance(x, o1)[0]
        d2 = point_object_distance(x, o2)[0]
        return -np.array([d1, d2]) + 1e-6

    res = minimize(
        cost,
        jac=jac,
        x0=x0,
        method="slsqp",
        constraints=[{"type": "ineq", "fun": ineq_con}],
    )
    if not res.success:
        raise ValueError("Optimizer did not converge!")
    return res.x


def find_candidate_vertices(o1, o2, n, x0, tol=1e-3):
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    c = np.cross(n, x)
    if np.linalg.norm(c) <= tol:
        c = np.cross(n, y)

    c1 = c / np.linalg.norm(c)
    c2 = np.cross(n, c1)
    c3 = (c1 + c2) / np.linalg.norm(c1 + c2)
    c4 = np.cross(n, c3)

    C = np.vstack((c1, -c1, c2, -c2, c3, -c3, c4, -c4))
    points = []
    for i in range(C.shape[0]):
        p = solve_vertex(o1, o2, C[i, :], x0)
        points.append(p)
    return np.array(points)


def two_object_opt(o1, o2, tol=1e-3):
    d, x0, normal = nearest_distance(o1, o2)
    if d > tol:
        print("not in contact")
        return [], normal

    X = find_candidate_vertices(o1, o2, normal, x0, tol=tol)

    # remove points that are too close together
    D = squareform(pdist(X))
    unique = 1 - np.sum(np.tril(D < 1e-3, k=-1), axis=1)
    unique_idx = np.nonzero(unique)[0]

    # only the unique points
    X = X[unique_idx, :]
    D = D[unique_idx, :][:, unique_idx]

    # if there is only one or two (necessarily unique) points left, they are
    # the vertices
    if X.shape[0] <= 2:
        return X, normal

    # the two points farthest apart are definitely vertices
    idx = np.unravel_index(np.argmax(D), D.shape)

    # the other one or two vertices are the remaining points that maximize and
    # minimize, respectively, the distance to the line connecting the two
    # current vertices
    x0 = X[idx[0], :]
    x1 = X[idx[1], :]
    a = np.cross(normal, x1 - x0)
    a = a / np.linalg.norm(a)

    proj = (X - x0) @ a
    i1 = np.argmax(proj)
    i2 = np.argmin(proj)

    # if either distance is too small, we don't include the vertex because it
    # is just on the line between the two existing ones
    indices = list(idx)
    if np.abs(proj[i1]) > tol:
        indices.append(i1)
    if np.abs(proj[i2]) > tol:
        indices.append(i2)

    return X[indices, :], normal


def box(side_lengths, position, color):
    side_lengths = np.array(side_lengths)
    position = np.array(position)

    box = simulation.BulletBody.cuboid(MASS, MU, side_lengths, color=color)
    box.add_to_sim(position)
    return fcl.CollisionObject(fcl.Box(*side_lengths), fcl.Transform3f(position))


def cylinder(radius, height, position, color):
    position = np.array(position)

    cy = simulation.BulletBody.cylinder(MASS, MU, radius, height, color=color)
    cy.add_to_sim(position)
    return fcl.CollisionObject(fcl.Cylinder(radius, height), fcl.Transform3f(position))


def box_example():
    o1 = box([0.2, 0.2, 0.2], [0, 0, 0.1], color=(1, 0, 0, 0.5))
    o2 = box([0.2, 0.2, 0.2], [0.3, 0.1, 0.1], color=(0, 0, 1, 0.5))
    o3 = box([0.4, 0.2, 0.1], [0.2, 0, 0.25], color=(0, 1, 0, 0.5))

    vs2, _ = two_object_opt(o1, o3)
    vs3, _ = two_object_opt(o2, o3)

    return np.vstack((vs2, vs3))


def cylinder_example():
    o1 = box([0.2, 0.2, 0.2], [0, 0, 0.1], color=(1, 0, 0, 0.5))
    o2 = box([0.2, 0.2, 0.2], [0.2, 0.1, 0.1], color=(0, 0, 1, 0.5))
    o3 = cylinder(0.1, 0.2, [0, 0.2, 0.1], color=(0, 1, 0, 0.5))

    vs1, n = two_object_opt(o1, o2)
    vs2, _ = two_object_opt(o1, o3)
    vs3, _ = two_object_opt(o2, o3)

    return np.vstack((vs1, vs2, vs3))


def main():
    np.set_printoptions(precision=8, suppress=True)

    pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    pyb.resetDebugVisualizerCamera(
        cameraDistance=4,
        cameraYaw=42,
        cameraPitch=-35.8,
        cameraTargetPosition=[1.28, 0.045, 0.647],
    )
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)

    # vs = box_example()
    # vs = cylinder_example()

    # colors = [[0, 0, 0] for _ in vs]
    # pyb.addUserDebugPoints(vs, colors, pointSize=10)

    IPython.embed()


if __name__ == "__main__":
    main()
