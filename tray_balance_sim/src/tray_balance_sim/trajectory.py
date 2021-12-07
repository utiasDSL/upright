import numpy as np


class LinearTimeScaling:
    ''' Linear time-scaling: constant velocity. '''
    def __init__(self, duration):
        self.duration = duration

    def eval(self, t):
        s = t / self.duration
        ds = np.ones_like(t) / self.duration
        dds = np.zeros_like(t)
        return s, ds, dds


class CubicTimeScaling:
    ''' Cubic time-scaling: zero velocity at end points. '''
    def __init__(self, duration):
        self.coeffs = np.array([0, 0, 3 / duration**2, -2 / duration**3])

    def eval(self, t):
        s = self.coeffs.dot([np.ones_like(t), t, t**2, t**3])
        ds = self.coeffs[1:].dot([np.ones_like(t), 2*t, 3*t**2])
        dds = self.coeffs[2:].dot([2*np.ones_like(t), 6*t])
        return s, ds, dds


class QuinticTimeScaling:
    ''' Quintic time-scaling: zero velocity and acceleration at end points. '''
    def __init__(self, T):
        A = np.array([[1, 0, 0, 0, 0, 0],
                      [1, T, T**2, T**3, T**4, T**5],
                      [0, 1, 0, 0, 0, 0],
                      [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
                      [0, 0, 2, 0, 0, 0],
                      [0, 0, 2, 6*T, 12*T**2, 20*T**3]])
        b = np.array([0, 1, 0, 0, 0, 0])
        self.coeffs = np.linalg.solve(A, b)

    def eval(self, t):
        s = self.coeffs.dot([np.ones_like(t), t, t**2, t**3, t**4, t**5])
        ds = self.coeffs[1:].dot([np.ones_like(t), 2*t, 3*t**2, 4*t**3, 5*t**4])
        dds = self.coeffs[2:].dot([2*np.ones_like(t), 6*t, 12*t**2, 20*t**3])
        return s, ds, dds


def eval_trapezoidal(t, v, a, ta, T):
    ''' Evaluate trapezoidal time-scaling given:
        t:  evaluation times
        v:  cruise velocity
        a:  acceleration
        ta: time of acceleration
        T:  total duration '''
    t1 = t < ta
    t2 = (t >= ta) & (t < T - ta)
    t3 = t >= T - ta

    # stage 1: acceleration
    s1 = 0.5*a*t**2
    ds1 = a*t
    dds1 = a * np.ones_like(t)

    # stage 2: constant velocity
    s2 = v*t - 0.5*v**2/a
    ds2 = v * np.ones_like(t)
    dds2 = np.zeros_like(t)

    # stage 3: deceleration
    s3 = v*T - v**2/a - 0.5*a*(t - T)**2
    ds3 = a*(T - t)
    dds3 = -a * np.ones_like(t)

    s = t1 * s1 + t2 * s2 + t3 * s3
    ds = t1 * ds1 + t2 * ds2 + t3 * ds3
    dds = t1 * dds1 + t2 * dds2 + t3 * dds3

    return s, ds, dds


class TrapezoidalTimeScalingV:
    ''' Trapezoidal time scaling specifying cruising velocity and duration.
        v should be such that 2 >= v*T > 1 for a 3-stage profile. '''
    def __init__(self, v, duration):
        self.v = v
        self.duration = duration
        self.a = v**2 / (v*duration - 1)
        self.ta = v / self.a

    def eval(self, t):
        return eval_trapezoidal(t, self.v, self.a, self.ta, self.duration)


class TrapezoidalTimeScalingA:
    ''' Trapezoidal time scaling specifying acceleration and duration.
        a should be such that a*T**2 >= 4 to ensure the motion is completed in
        time. '''
    def __init__(self, a, duration):
        self.v = 0.5*(a*duration - np.sqrt(a*(a*duration**2 - 4)))
        self.duration = duration
        self.a = a
        self.ta = self.v / a

    def eval(self, t):
        return eval_trapezoidal(t, self.v, self.a, self.ta, self.duration)


# == Paths == #


class Chain:
    ''' Chain multiple independent trajectories together. '''
    def __init__(self, trajectories):
        self.trajectories = trajectories

        durations = np.array([traj.duration for traj in trajectories])
        self.times = np.zeros(1 + durations.shape[0])
        self.times[1:] = np.cumsum(durations)

    def sample(self, t, flatten=False):
        samples = []

        for idx, trajectory in enumerate(self.trajectories):
            t0 = self.times[idx]  # start
            t1 = self.times[idx+1]  # end

            # all times beyond the total duration are passed to the last
            # trajectory, which is expected to handle it
            if idx == len(self.trajectories) - 1:
                ts = t[(t >= t0)]
            else:
                ts = t[(t >= t0) & (t < t1)]
            samples.append(trajectory.sample(ts - t0, flatten=flatten))

        p = np.concatenate([sample[0] for sample in samples])
        v = np.concatenate([sample[1] for sample in samples])
        a = np.concatenate([sample[2] for sample in samples])

        return p, v, a


class CubicBezier:
    ''' Cubic Bezier curve trajectory. '''
    def __init__(self, points, timescaling, duration):
        ''' Points should be a (4*2) array of control points, with p[0, :]
            being the initial position and p[-1, :] the final. '''
        self.points = points
        self.timescaling = timescaling
        self.duration = duration

    def sample(self, t, flatten=False):
        s, ds, dds = self.timescaling.eval(t)

        cp = np.array([(1-s)**3, 3*(1-s)**2*s, 3*(1-s)*s**2, s**3])
        p = cp.T.dot(self.points)

        cv = np.array([3*(1-s)**2, 6*(1-s)*s, 3*s**2])
        dpds = cv.T.dot(self.points[1:, :] - self.points[:-1, :])
        v = (dpds.T * ds).T

        ca = np.array([6*(1-s), 6*s])
        dpds2 = ca.T.dot(self.points[2:, :] - 2*self.points[1:-1, :] + self.points[:-2, :])
        a = (dpds.T * dds + dpds2.T * ds**2).T

        if flatten:
            return p.flatten(), v.flatten(), a.flatten()

        return p, v, a


class PointToPoint:
    ''' Point-to-point trajectory. '''
    def __init__(self, p0, p1, timescaling, duration):
        self.p0 = p0
        self.p1 = p1
        self.timescaling = timescaling
        self.duration = duration

    def sample(self, t, flatten=False):
        s, ds, dds = self.timescaling.eval(t)
        p = self.p0 + (s * (self.p1 - self.p0)[:, None]).T
        v = (ds * (self.p1 - self.p0)[:, None]).T
        a = (dds * (self.p1 - self.p0)[:, None]).T
        if flatten:
            return p.flatten(), v.flatten(), a.flatten()
        return p, v, a


class Circle:
    ''' Circular trajectory. '''
    def __init__(self, p0, r, timescaling, duration):
        ''' p0 is the starting point; center is p0 + [r, 0] '''
        self.r = r
        self.pc = p0 + np.array([r, 0])  # start midway up left side of circle
        self.timescaling = timescaling
        self.duration = duration

    def sample(self, t, flatten=False):
        s, ds, dds = self.timescaling.eval(t)

        cs = np.cos(2*np.pi*s - np.pi)
        ss = np.sin(2*np.pi*s - np.pi)
        p = self.pc + self.r * np.array([cs, ss]).T

        dpds = 2*np.pi*self.r * np.array([-ss, cs]).T
        v = dpds * ds[:, None]

        dpds2 = 4*np.pi**2*self.r * np.array([-cs, -ss]).T
        a = dpds * dds[:, None] + dpds2 * ds[:, None]**2

        if flatten:
            return p.flatten(), v.flatten(), a.flatten()
        return p, v, a


class Sine:
    ''' Sinusoidal trajectory: linear in x, sinusoidal in y. '''
    def __init__(self, p0, lx, amp, freq, timescaling, duration):
        self.p0 = p0
        self.lx = lx
        self.A = amp

        # multiply by 2*pi so that sin(s) = sin(2*pi) at s = 1
        self.w = freq * 2 * np.pi

        self.timescaling = timescaling
        self.duration = duration

    def sample(self, t, flatten=False):
        s, ds, dds = self.timescaling.eval(t)

        # x is linear in s
        x = self.p0[0] + self.lx * s
        dx = self.lx * ds
        ddx = self.lx * dds

        # y = A*sin(w*s)
        y = self.p0[1] + self.A*np.sin(self.w*s)
        dyds = self.A*self.w*np.cos(self.w*s)
        dyds2 = -self.w**2 * y
        dy = dyds * ds
        ddy = dyds * dds + dyds2 * ds**2

        p = np.vstack((x, y)).T
        v = np.vstack((dx, dy)).T
        a = np.vstack((ddx, ddy)).T

        if flatten:
            return p.flatten(), v.flatten(), a.flatten()
        return p, v, a


class Point:
    ''' Stationary point trajectory. '''
    def __init__(self, p0):
        self.p0 = p0

    def sample(self, t, flatten=False):
        if np.isscalar(t):
            p = self.p0
        else:
            p = np.tile(self.p0, (t.shape[0], 1))
        v = np.zeros_like(p)
        a = np.zeros_like(p)
        if flatten:
            return p.flatten(), v.flatten(), a.flatten()
        return p, v, a
