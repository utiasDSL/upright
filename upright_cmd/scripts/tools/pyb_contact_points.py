from upright_sim import simulation
import numpy as np
import pybullet as pyb
import hppfcl as fcl
import scipy
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
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


def nearest_distance(o1, o2):
    req = fcl.DistanceRequest()
    res = fcl.DistanceResult()
    dist = fcl.distance(o1, o2, req, res)
    return dist, res.getNearestPoint2()


def two_object_opt(s1, s2, T1, T2):
    c1 = T1.transform(np.zeros(3))
    c2 = T2.transform(np.zeros(3))
    o1 = s1.collision_object(T1)
    o2 = s2.collision_object(T2)

    d, p = nearest_distance(o1, o2)
    if d > 1e-6:
        print("not in contact")
        return []

    def cost(x, ps):
        n = ps.shape[0]
        # cost1 = -0.5 * n * x[:3] @ x[:3] + np.sum(ps @ x[:3]) - 0.5 * np.sum(ps * ps)
        # cost2 = -0.5 * n * x[3:] @ x[3:] + np.sum(ps @ x[3:]) - 0.5 * np.sum(ps * ps)
        # return cost1 + cost2

        return -0.5 * n * x @ x + np.sum(ps @ x) - np.sum(ps * ps)
        # Δ = x - p
        # return -0.5 * Δ @ Δ

    def jac(x, ps):
        n = ps.shape[0]
        # J1 = -n * x[:3] + np.sum(ps, axis=0)
        # J2 = -n * x[3:] + np.sum(ps, axis=0)
        # J = np.concatenate((J1, J2))
        # print(f"ps = {ps}")
        # print(f"J = {J}")
        # return J
        return -n * x + np.sum(ps, axis=0)
        # return -0.5 * x + p

    def ineq_con(x):
        # d1, _ = point_object_distance(x, o1)
        # d2, _ = point_object_distance(x, o2)
        # return -np.array([d1, d2]) + 1e-6
        # con1 = s1.constraints(x[:3], c1)
        # con2 = s2.constraints(x[3:], c2)
        # return np.concatenate((con1, con2)) + 1e-6
        con1 = s1.constraints(x, c1)
        con2 = s2.constraints(x, c2)
        return np.concatenate((con1, con2))

    def ineq_con_jac(x):
        J1 = s1.constraints_jac(x, c1)
        J2 = s2.constraints_jac(x, c2)
        return np.vstack((J1, J2))
        # J1 = s1.constraints_jac(x[:3], c1)
        # J2 = s2.constraints_jac(x[3:], c2)
        # J = np.zeros((12, 6))
        # J[:6, :3] = J1
        # J[6:, 3:] = J2
        # return J

    def eq_con(x):
        return x[:3] - x[3:]

    ps = p[None, :]
    res = minimize(
        cost,
        x0=p + np.random.random(3),
        # x0=np.concatenate((ps[0, :], ps[0, :])),
        args=(ps,),
        method="slsqp",
        jac=jac,
        constraints=[{"type": "ineq", "fun": ineq_con, "jac": ineq_con_jac}],
        # constraints=[{"type": "ineq", "fun": ineq_con, "jac": ineq_con_jac}, {"type": "eq", "fun": eq_con}],
    )

    print("one")

    ps = res.x[None, :3]
    res = minimize(
        cost,
        x0=p,
        # x0=np.concatenate((ps[0, :], ps[0, :])) + 0.1 * np.random.random(6),
        args=(ps,),
        method="slsqp",
        jac=jac,
        constraints=[{"type": "ineq", "fun": ineq_con, "jac": ineq_con_jac}],
        # constraints=[{"type": "ineq", "fun": ineq_con, "jac": ineq_con_jac}, {"type": "eq", "fun": eq_con}],
    )

    # print("two")
    #
    # ps = np.vstack((ps, res.x[:3]))
    # res = minimize(
    #     cost,
    #     # x0=ps[0, :],
    #     x0=np.concatenate((ps[0, :], ps[0, :])) + 0.1 * np.random.random(6),
    #     args=(ps,),
    #     method="slsqp",
    #     jac=jac,
    #     # constraints=[{"type": "ineq", "fun": ineq_con, "jac": ineq_con_jac}],
    #     constraints=[{"type": "ineq", "fun": ineq_con, "jac": ineq_con_jac}, {"type": "eq", "fun": eq_con}],
    # )

    IPython.embed()


def main():
    np.set_printoptions(precision=3, suppress=True)

    pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    pyb.resetDebugVisualizerCamera(
        cameraDistance=4,
        cameraYaw=42,
        cameraPitch=-35.8,
        cameraTargetPosition=[1.28, 0.045, 0.647],
    )
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)

    box1 = simulation.BulletBody.cuboid(MASS, MU, [0.2, 0.2, 0.2], color=(1, 0, 0, 0.5))
    box2 = simulation.BulletBody.cuboid(MASS, MU, [0.2, 0.2, 0.2], color=(0, 0, 1, 0.5))
    cy1 = simulation.BulletBody.cylinder(MASS, MU, 0.1, 0.2, color=(0, 1, 0, 0.5))

    box1.add_to_sim([0, 0, 0.1])
    box2.add_to_sim([0.2, 0, 0.1])
    cy1.add_to_sim([0, 0.2, 0.1])

    o1 = fcl.CollisionObject(
        fcl.Box(0.2, 0.2, 0.2), fcl.Transform3f(np.array([0, 0, 0.1]))
    )
    o2 = fcl.CollisionObject(
        fcl.Box(0.2, 0.2, 0.2), fcl.Transform3f(np.array([0.2, 0, 0.1]))
    )

    box1 = Box([0.2, 0.2, 0.2])
    box2 = Box([0.2, 0.2, 0.2])
    cy1 = Cylinder(0.1, 0.2)
    two_object_opt(
        box1,
        box2,
        fcl.Transform3f(np.array([0, 0, -0.1])),
        fcl.Transform3f(np.array([0.2, 0, -0.1])),
    )
    # two_object_contact_points(box1, cy1, fcl.Transform3f(np.array([0, 0, 0.1])), fcl.Transform3f(np.array([0.2, 0, 0.1])))
    return

    request = fcl.CollisionRequest(fcl.CONTACT, 10000)
    result = fcl.CollisionResult()
    ret = fcl.collide(o1, o2, request, result)
    print(ret)

    pyb.performCollisionDetection()
    contact_points = pyb.getContactPoints(
        # box1.uid,
        # cy1.uid,
    )
    # TODO: can we special case the (upright) cylinder?
    # find a contact point, then search the line in the z-direction from top to
    # bottom of the cylinder
    # but we'd also like to handle a cylinder on top of another object
    # can I actually replace it with cuboids to find the points?
    xs = [p[5] for p in contact_points]
    # for p in contact_points:
    #     if p[1] == cy1 or p[2] == cy1:
    #         x = p[5]
    #         IPython.embed()
    colors = [[0, 0, 0] for _ in contact_points]

    pyb.addUserDebugPoints(xs, colors, pointSize=10)

    IPython.embed()


if __name__ == "__main__":
    main()
