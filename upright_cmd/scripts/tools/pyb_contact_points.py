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


def two_object_contact_points(s1, s2, T1, T2):
    vs1 = s1.vertices(T1)
    vs2 = s2.vertices(T2)
    o1 = s1.collision_object(T1)
    o2 = s2.collision_object(T2)

    candidate_points = []

    for i in range(vs1.shape[0]):
        # get nearest point on other object to this vertex
        _, q = point_object_distance(vs1[i, :], o2)

        # get distance of the nearest point to the original object
        d, _ = point_object_distance(q, o1)

        # only add the nearest point if it is also located on the original
        # object
        if d < 1e-6:
            candidate_points.append(q)

    # same thing but for the other object
    for i in range(vs2.shape[0]):
        _, q = point_object_distance(vs2[i, :], o1)
        d, _ = point_object_distance(q, o2)
        print(q)
        print(d)
        if d < 1e-6:
            candidate_points.append(q)

    IPython.embed()

    # project into a lower-dimensional space to be full rank: contact manifold
    # will always be less than 3 for convex objects in 3D space
    # this is needed for the convex hull algorithm
    A = np.array(candidate_points)
    c = np.mean(A, axis=0)
    r = np.linalg.matrix_rank(A - c)
    _, _, V = scipy.linalg.svd(A - c)
    X = (A @ V.T)[:, :r]

    # only take the points that are at the vertices of the constraint manifold
    hull = ConvexHull(X)
    points = A[hull.vertices]

    IPython.embed()
    return points


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


def solve_lp(o1, o2, c, x0):
    C = np.concatenate((c, -c))
    def cost(x):
        return C @ x

    def jac(x):
        return C

    def ineq_con(x):
        d11 = point_object_distance(x[:3], o1)[0]
        d12 = point_object_distance(x[:3], o2)[0]
        d21 = point_object_distance(x[3:], o1)[0]
        d22 = point_object_distance(x[3:], o2)[0]
        return -np.array([d11, d12, d21, d22]) + 1e-6

    res = minimize(
        cost,
        jac=jac,
        x0=np.concatenate((x0, x0)),
        method="slsqp",
        constraints=[{"type": "ineq", "fun": ineq_con}],
    )
    return res.x.reshape((2, 3))



def two_object_opt(o1, o2):
    d, p, n = nearest_distance(o1, o2)
    if d > 1e-6:
        print("not in contact")
        return [], n

    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    c = np.cross(n, x)
    if np.linalg.norm(c) <= 1e-3:
        c = np.cross(n, y)

    # first find eight candidate vertices (they are all points on the boundary)
    # TODO we could solve this is a single optimization, since all the C are
    # computed beforehand
    c1 = c / np.linalg.norm(c)
    X1 = solve_lp(o1, o2, c1, p)

    c2 = np.cross(n, c1)
    X2 = solve_lp(o1, o2, c2, p)

    c3 = (c1 + c2) / np.linalg.norm(c1 + c2)
    X3 = solve_lp(o1, o2, c3, p)

    c4 = np.cross(n, c3)
    X4 = solve_lp(o1, o2, c4, p)

    X = np.vstack((X1, X2, X3, X4))

    # the two points farthest apart are definitely vertices (unless they are
    # the same point)
    D = squareform(pdist(X))
    idx = np.unravel_index(np.argmax(D), D.shape)
    max_dist = D[idx]

    tol = 1e-3
    if max_dist < tol:
        indices = [idx[0]]
    else:
        indices = list(idx)
        ds = np.sum(D[:, idx], axis=1)
        other_idx = np.argpartition(ds, 2)[-2:]

        for i in range(2):
            if ds[other_idx[i]] > max_dist + tol:
                indices.append(other_idx[i])

    vertices = X[indices, :]

    return vertices, n


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

    # o1 = box([0.2, 0.2, 0.2], [0, 0, 0.1], color=(1, 0, 0, 0.5))
    # o2 = box([0.2, 0.2, 0.2], [0.2, 0.1, 0.1], color=(0, 0, 1, 0.5))
    # o3 = cylinder(0.1, 0.2, [0, 0.2, 0.1], color=(0, 1, 0, 0.5))
    #
    # vs1, n = two_object_opt(o1, o2)
    # vs2, _ = two_object_opt(o1, o3)
    # vs3, _ = two_object_opt(o2, o3)

    o1 = box([0.2, 0.2, 0.2], [0, 0, 0.1], color=(1, 0, 0, 0.5))
    o2 = box([0.2, 0.2, 0.2], [0.3, 0.1, 0.1], color=(0, 0, 1, 0.5))
    o3 = box([0.4, 0.2, 0.1], [0.2, 0, 0.25], color=(0, 1, 0, 0.5))

    vs1, _ = two_object_opt(o1, o2)
    vs2, _ = two_object_opt(o1, o3)
    vs3, _ = two_object_opt(o2, o3)

    vs = np.vstack((vs2, vs3))

    colors = [[0, 0, 0] for _ in vs]
    pyb.addUserDebugPoints(vs, colors, pointSize=10)

    IPython.embed()


if __name__ == "__main__":
    main()
