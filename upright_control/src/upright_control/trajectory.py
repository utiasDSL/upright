import numpy as np
from scipy.linalg import expm

import IPython


class DoubleIntegrator:
    def __init__(self, n):
        self.n = n

        Z = np.zeros((n, n))
        I = np.eye(n)
        A = np.block([[Z, I], [Z, Z]])
        B = np.vstack((Z, I))

        self.M = np.block([[A, B], [np.zeros((n, 2*n)), Z]])

    def integrate(self, v, a, u, dt):
        # TODO for some reason this causes non-smooth behaviour
        x = np.concatenate((v, a))
        Md = expm(dt * self.M)
        Ad = Md[:2*self.n, :2*self.n]
        Bd = Md[:2*self.n, 2*self.n:]
        x_new = Ad @ x + Bd @ u

        v = x_new[:self.n]
        a = x_new[self.n:]
        return v, a

    def integrate_approx(self, v, a, u, dt):
        a_new = a + dt * u
        v_new = v + dt * a_new
        return v_new, a_new



class StateInputTrajectory:
    """Generic state-input trajectory."""
    def __init__(self, ts, xs, us):
        assert len(ts) == len(xs) == len(us)

        self.ts = ts
        self.xs = xs
        self.us = us

    @classmethod
    def load(cls, filename):
        with np.load(filename) as data:
            ts = data["ts"]
            xs = data["xs"]
            us = data["us"]
        return cls(ts=ts, xs=xs, us=us)

    def save(self, filename):
        np.savez_compressed(filename, ts=self.ts, xs=self.xs, us=self.us)

    def __getitem__(self, idx):
        return self.ts[idx], self.xs[idx], self.us[idx]

    def __len__(self):
        return len(self.ts)


class StateInputMapping:
    """Mapping to/from (state, input) and (position, velocity, acceleration)."""
    def __init__(self, dims):
        self.dims = dims

    def xu2qva(self, x, u=None):
        q = x[: self.dims.q]
        v = x[self.dims.q : self.dims.q + self.dims.v]
        a = x[self.dims.q + self.dims.v : self.dims.q + 2 * self.dims.v]
        return q, v, a

    def qva2xu(self, q, v, a):
        x = np.concatenate((q, v, a))
        return x, None


class QuinticPoint:
    def __init__(self, t, q, v, a):
        self.t = t
        self.q = q
        self.v = v
        self.a = a


class QuinticInterpolator:
    def __init__(self, p1, p2):
        self.t0 = p1.t
        T = p2.t - p1.t
        A = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [1, T, T ** 2, T ** 3, T ** 4, T ** 5],
                [0, 1, 0, 0, 0, 0],
                [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                [0, 0, 2, 0, 0, 0],
                [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3],
            ]
        )
        b = np.array([p1.q, p2.q, p1.v, p2.v, p1.a, p2.a])

        # it is actually much faster to loop and solve the vector systems
        # rather than one big matrix system
        self.coeffs = np.zeros_like(b)
        for i in range(b.shape[1]):
            self.coeffs[:, i] = np.linalg.solve(A, b[:, i])

    def interpolate(self, t):
        t = np.array(t) - self.t0  # normalize time

        zs = np.zeros_like(t)
        os = np.ones_like(t)

        w = np.array([os, t, t ** 2, t ** 3, t ** 4, t ** 5])
        dw = np.array([zs, os, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4])
        ddw = np.array([zs, zs, 2 * os, 6 * t, 12 * t ** 2, 20 * t ** 3])

        q = w.T @ self.coeffs
        v = dw.T @ self.coeffs
        a = ddw.T @ self.coeffs

        return q, v, a


class LinearInterpolator:
    def __init__(self, t1, q1, t2, q2):
        self.t0 = t1

        Δt = t2 - t1
        A = np.array([[1, 0], [1, Δt]])
        b = np.array([q1, q2])
        self.coeffs = np.linalg.solve(A, b)

    def interpolate(self, t):
        t = np.array(t) - self.t0  # normalize time
        w = np.array([np.ones_like(t), t])
        q = w.T @ self.coeffs
        return q


class TrajectoryInterpolator:
    def __init__(self, mapping, trajectory):
        self.mapping = mapping
        self.update(trajectory)

    def update(self, trajectory):
        self.trajectory = trajectory

    def interpolate(self, t):
        # make sure we are in the correct time range
        # if t < self.trajectory.ts[self.traj_idx]:
        #     raise ValueError(
        #         f"We are going back in time with t = {t} but our trajectory is at t = {self.trajectory.ts[self.traj_idx]}"
        #     )
        if t > self.trajectory.ts[-1]:
            raise ValueError(
                f"We are at t = {t} but our trajectory only goes to t = {self.trajectory.ts[-1]}"
            )

        # if we are before the trajectory, then just return the first element
        # (we shouldn't be much before it)
        if t <= self.trajectory.ts[0]:
            _, x, u = self.trajectory[0]
            return x

        # find the new point in the trajectory
        for i in range(len(self.trajectory) - 1):
            if self.trajectory.ts[i] <= t <= self.trajectory.ts[i + 1]:
                break

        # TODO we could only update the interpolator if we are between new
        # waypoints, but this is more complex

        t1, x1, u1 = self.trajectory[i]
        t2, x2, u2 = self.trajectory[i + 1]

        # if the timestep is small, just steer toward the end waypoint
        if t2 - t1 <= 1e-3:
            return x2

        # if we need to do interpolation, and we're between two new waypoints
        # in the trajectory, then update the interpolator
        q1, v1, a1 = self.mapping.xu2qva(x1, u1)
        p1 = QuinticPoint(t=t1, q=q1, v=v1, a=a1)

        q2, v2, a2 = self.mapping.xu2qva(x2, u2)
        p2 = QuinticPoint(t=t2, q=q2, v=v2, a=a2)

        # do interpolation
        qd, vd, ad = QuinticInterpolator(p1, p2).interpolate(t)
        xd = self.mapping.qva2xu(qd, vd, ad)[0]
        return xd
