#!/usr/bin/env python3
"""Plot end effector position, velocity, and acceleration from a bag file.

Also computes the maximum velocity and acceleration.
"""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import signal
import seaborn

from mobile_manipulation_central import ros_utils
import upright_core as core

import IPython


VICON_OBJECT_NAME = "ThingWoodTray"


def savgol(x, y, window_length, polyorder, deriv=0):
    """Savgol filter for non-uniformly spaced data.

    Much slower than scipy.signal.savgol_filter because we need to refit the
    polynomial to every window of data.
    """
    assert len(x) == len(y)
    assert window_length > polyorder
    assert type(deriv) == int and deriv >= 0

    r = window_length // 2
    n = len(x)
    degree = polyorder - deriv

    y_smooth = np.zeros_like(y)
    for i in range(n):
        low = max(i - r, 0)
        high = min(i + r + 1, n)
        x_window = x[low:high]
        y_window = y[low:high]
        poly = np.polynomial.Polynomial.fit(x_window, y_window, polyorder)
        for _ in range(deriv):
            poly = poly.deriv()
        y_smooth[i] = poly(x[i])
    return y_smooth


class FilterUpdater:
    """Add sliders to a plot to control SavGol filter window size and poly order."""
    def __init__(
        self, fig, lines, positions, window_size, polyorder, delta=1.0, deriv=0
    ):
        self.fig = fig
        self.lines = lines
        self.positions = positions
        self.window_size = window_size
        self.polyorder = polyorder
        self.delta = delta
        self.deriv = deriv

        ax_window_slider = fig.add_axes([0.35, 0.15, 0.5, 0.03])
        ax_poly_slider = fig.add_axes([0.35, 0.1, 0.5, 0.03])

        self.window_slider = Slider(
            ax_window_slider, "Window Size", 5, 100, valinit=window_size, valstep=1
        )
        self.poly_slider = Slider(
            ax_poly_slider, "Poly Order", 1, 10, valinit=polyorder, valstep=1
        )

        self.window_slider.on_changed(self.update_window_size)
        self.poly_slider.on_changed(self.update_polyorder)

    def update(self):
        y = signal.savgol_filter(
            self.positions,
            self.window_size,
            self.polyorder,
            deriv=self.deriv,
            delta=self.delta,
            axis=0,
        )
        for i, line in enumerate(self.lines[:3]):
            line.set_ydata(y[:, i])
        self.lines[-1].set_ydata(np.linalg.norm(y, axis=1))
        self.fig.canvas.draw_idle()

    def update_window_size(self, window_size):
        self.window_size = window_size
        self.update()

    def update_polyorder(self, polyorder):
        self.polyorder = polyorder
        self.update()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()

    bag = rosbag.Bag(args.bagfile)

    topic = ros_utils.vicon_topic_name(VICON_OBJECT_NAME)
    msgs = [msg for _, msg, _ in bag.read_messages(topic)]
    ts, poses = ros_utils.parse_transform_stamped_msgs(msgs, normalize_time=True)

    n = len(ts)
    positions = poses[:, :3]
    dt = np.mean(ts[1:] - ts[:-1])  # average time step

    # smoothed velocities and accelerations using Savitzky-Golay filter
    window_size = 31
    polyorder = 2
    smooth_velocities = signal.savgol_filter(
        positions, window_size, polyorder, deriv=1, delta=dt, axis=0
    )
    smooth_accelerations = signal.savgol_filter(
        positions, window_size, polyorder, deriv=2, delta=dt, axis=0
    )

    # (noisy) velocities and accelerations computed by finite differences
    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)
    for i in range(1, n - 1):
        velocities[i, :] = (positions[i + 1, :] - positions[i - 1, :]) / (
            ts[i + 1] - ts[i - 1]
        )

    for i in range(2, n - 2):
        accelerations[i, :] = (velocities[i + 1, :] - velocities[i - 1, :]) / (
            ts[i + 1] - ts[i - 1]
        )

    palette = seaborn.color_palette("deep")

    plt.figure()
    plt.plot(ts, positions[:, 0], label="x")
    plt.plot(ts, positions[:, 1], label="y")
    plt.plot(ts, positions[:, 2], label="z")
    plt.title("EE position")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.legend()
    plt.grid()

    vel_fig = plt.figure()
    plt.plot(ts, velocities[:, 0], color=palette[0], alpha=0.2)
    plt.plot(ts, velocities[:, 1], color=palette[1], alpha=0.2)
    plt.plot(ts, velocities[:, 2], color=palette[2], alpha=0.2)
    plt.plot(ts, np.linalg.norm(velocities, axis=1), color=palette[3], alpha=0.2)
    (l1,) = plt.plot(ts, smooth_velocities[:, 0], label="x", color=palette[0])
    (l2,) = plt.plot(ts, smooth_velocities[:, 1], label="y", color=palette[1])
    (l3,) = plt.plot(ts, smooth_velocities[:, 2], label="z", color=palette[2])
    (l4,) = plt.plot(ts, np.linalg.norm(smooth_velocities, axis=1), label="norm", color=palette[3])
    plt.title("EE velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend()
    plt.grid()

    vel_updater = FilterUpdater(
        vel_fig, [l1, l2, l3, l4], positions, window_size, polyorder, delta=dt, deriv=1
    )

    acc_fig = plt.figure()
    plt.plot(ts, accelerations[:, 0], color=palette[0], alpha=0.2)
    plt.plot(ts, accelerations[:, 1], color=palette[1], alpha=0.2)
    plt.plot(ts, accelerations[:, 2], color=palette[2], alpha=0.2)
    plt.plot(ts, np.linalg.norm(accelerations, axis=1), color=palette[3], alpha=0.2)
    (l1,) = plt.plot(ts, smooth_accelerations[:, 0], label="x", color=palette[0])
    (l2,) = plt.plot(ts, smooth_accelerations[:, 1], label="y", color=palette[1])
    (l3,) = plt.plot(ts, smooth_accelerations[:, 2], label="z", color=palette[2])
    (l4,) = plt.plot(ts, np.linalg.norm(smooth_accelerations, axis=1), label="norm", color=palette[3])
    plt.title("EE acceleration")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s]")
    plt.legend()
    plt.grid()

    acc_updater = FilterUpdater(
        acc_fig, [l1, l2, l3, l4], positions, window_size, polyorder, delta=dt, deriv=2
    )

    plt.show()


if __name__ == "__main__":
    main()
