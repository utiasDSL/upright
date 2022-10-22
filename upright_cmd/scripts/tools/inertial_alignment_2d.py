import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import IPython


DT = 0.01
DURATION = 20
G = 9.81

t = 0
angle = 0
angle_dot = 0
x = 0
x_dot = 0

angles = [angle]
xs = [0]
ts = [t]

s = np.zeros(4)
sd = np.array([1, 0, 0, 0])
# k = np.array([100, 100, 100, 10])

# construct controllability matrix of closed-loop system
A = np.zeros((4, 4))
A[:-1, 1:] = np.eye(3)
B = np.array([[0], [0], [0], [1]])

# C is full rank, so system is controllable
C = np.hstack((B, A @ B, A @ A @ B, A @ A @ A @ B))

# Solve for k using LQR
Q = np.eye(4)
R = np.array([[1]])
P = scipy.linalg.solve_continuous_are(A, B, Q, R)
k = (B.T @ P / R[0, 0]).flatten()

while t < DURATION:
    u_prime = k @ (sd - s)
    u = u_prime * np.cos(angle)**2 / G - 2*np.tan(angle)*angle_dot**2

    # integrate angle
    angle_dot += DT * u
    angle += DT * angle_dot

    # integrate x
    x_dddot = G / np.cos(angle)**2 * angle_dot
    x_ddot = G * np.tan(angle)
    x_dot += DT * x_ddot
    x += DT * x_dot
    s = np.array([x, x_dot, x_ddot, x_dddot])

    t += DT
    angles.append(angle)
    ts.append(t)
    xs.append(x)

print(f"x = {xs[-1]}")
print(f"θ = {angles[-1]}")

plt.plot(ts, angles, label="θ")
plt.plot(ts, xs, label="x")
plt.legend()
plt.xlabel("Time (s)")
plt.grid()
plt.show()
