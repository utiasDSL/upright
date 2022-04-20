import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from liegroups import SO3
import yaml

import tray_balance_constraints as core

import IPython


class DataLogger:
    """Log data for later saving and viewing."""

    def __init__(self, config):
        self.config = config
        self.data = {}

    def add(self, key, value):
        """Add a single value named `key`."""
        if key in self.data:
            raise ValueError(f"Key {key} already in the data log.")
        self.data[key] = value

    def append(self, key, value):
        """Append a values to the list named `key`."""
        # copy to an array (also copies if value is already an array, which is
        # what we want)
        a = np.array(value)

        # append to list or start a new list if this is the first value under
        # `key`
        if key in self.data:
            if a.shape != self.data[key][-1].shape:
                raise ValueError("Data must all be the same shape.")
            self.data[key].append(a)
        else:
            self.data[key] = [a]

    def save(self, log_dir, timestamp, name=None):
        """Save the data and configuration to a timestamped directory."""
        dir_name = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        if name is not None:
            dir_name = name + "_" + dir_name
        dir_path = log_dir / dir_name
        dir_path.mkdir()

        data_path = dir_path / "data.npz"
        config_path = dir_path / "config.yaml"

        # save the recorded data
        np.savez_compressed(data_path, **self.data)

        # save the configuration used for this run
        with open(config_path, "w") as f:
            yaml.dump(self.config, stream=f, default_flow_style=False)

        print(f"Saved data to {dir_path}.")


class DataPlotter:
    def __init__(self, logger):
        # convert logger data to numpy format
        self.data = {}
        for key, value in logger.data.items():
            self.data[key] = np.array(value)

    def plot_ee_position(self):
        ts = self.data["ts"]
        r_ew_w_ds = self.data["r_ew_w_ds"]
        r_ew_ws = self.data["r_ew_ws"]

        plt.figure()
        plt.plot(ts, r_ew_w_ds[:, 0], label="$x_d$", color="r", linestyle="--")
        plt.plot(ts, r_ew_w_ds[:, 1], label="$y_d$", color="g", linestyle="--")
        plt.plot(ts, r_ew_w_ds[:, 2], label="$z_d$", color="b", linestyle="--")
        plt.plot(ts, r_ew_ws[:, 0], label="$x$", color="r")
        plt.plot(ts, r_ew_ws[:, 1], label="$y$", color="g")
        plt.plot(ts, r_ew_ws[:, 2], label="$z$", color="b")
        plt.grid()
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Position")
        plt.title("End effector position")

    def plot_ee_orientation(self):
        ts = self.data["ts"]
        Q_we_ds = self.data["Q_we_ds"]
        Q_wes = self.data["Q_wes"]

        plt.figure()
        plt.plot(ts, Q_we_ds[:, 0], label="$Q_{d,x}$", color="r", linestyle="--")
        plt.plot(ts, Q_we_ds[:, 1], label="$Q_{d,y}$", color="g", linestyle="--")
        plt.plot(ts, Q_we_ds[:, 2], label="$Q_{d,z}$", color="b", linestyle="--")
        plt.plot(ts, Q_wes[:, 0], label="$Q_{x}$", color="r")
        plt.plot(ts, Q_wes[:, 1], label="$Q_{y}$", color="g")
        plt.plot(ts, Q_wes[:, 2], label="$Q_{z}$", color="b")
        plt.grid()
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Orientation")
        plt.title("End effector orientation")

    def plot_ee_velocity(self):
        ts = self.data["ts"]
        v_ew_ws = self.data["v_ew_ws"]
        ω_ew_ws = self.data["ω_ew_ws"]

        plt.figure()
        plt.plot(ts, v_ew_ws[:, 0], label="$v_x$")
        plt.plot(ts, v_ew_ws[:, 1], label="$v_y$")
        plt.plot(ts, v_ew_ws[:, 2], label="$v_z$")
        plt.plot(ts, ω_ew_ws[:, 0], label="$ω_x$")
        plt.plot(ts, ω_ew_ws[:, 1], label="$ω_y$")
        plt.plot(ts, ω_ew_ws[:, 2], label="$ω_z$")
        plt.grid()
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity")
        plt.title("End effector velocity")

    def plot_object_error(self, obj_index):
        ts = self.data["ts"]
        r_ow_ws = self.data["r_ow_ws"][:, obj_index, :]
        r_ew_ws = self.data["r_ew_ws"]
        Q_wos = self.data["Q_wos"][:, obj_index, :]
        Q_wes = self.data["Q_wes"]
        obj_name = self.data["object_names"][obj_index]

        # object position error
        r_oe_es = np.zeros_like(r_ow_ws)
        for i in range(r_oe_es.shape[0]):
            C_ew = SO3.from_quaternion(Q_wes[i, :], ordering="xyzw").inv()
            r_oe_w = r_ow_ws[i, :] - r_ew_ws[i, :]
            r_oe_es[i, :] = C_ew.dot(r_oe_w)
        r_oe_e_err = r_oe_es - r_oe_es[0, :]

        plt.figure()
        plt.plot(ts, r_oe_e_err[:, 0], label="$x$")
        plt.plot(ts, r_oe_e_err[:, 1], label="$y$")
        plt.plot(ts, r_oe_e_err[:, 2], label="$z$")
        plt.plot(ts, np.linalg.norm(r_oe_e_err, axis=1), label="$||r||$")

        # object orientation error
        Q_ow0 = core.math.quat_inverse(Q_wos[0, :])
        Q_oe0 = core.math.quat_multiply(Q_ow0, Q_wes[0, :])
        Q_eo_err = np.zeros_like(Q_wos)
        angles = np.zeros(Q_wos.shape[0])
        for i in range(Q_eo_err.shape[0]):
            try:
                Q_ew = core.math.quat_inverse(Q_wes[i, :])
                Q_eo = core.math.quat_multiply(Q_ew, Q_wos[i, :])
                Q_eo_err[i, :] = core.math.quat_multiply(Q_oe0, Q_eo)
                angles[i] = core.math.quat_error(Q_eo_err[i, :])
            except ValueError as e:
                IPython.embed()

        plt.plot(ts, Q_eo_err[:, 0], label="$Q_x$")
        plt.plot(ts, Q_eo_err[:, 1], label="$Q_y$")
        plt.plot(ts, Q_eo_err[:, 2], label="$Q_z$")
        plt.plot(ts, angles, label=r"$\theta$")
        plt.grid()
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Error")
        plt.title(f"Object {obj_name} error")

    def plot_control_durations(self):
        durations = self.data["control_durations"]

        # the controller runs at a different rate than the simulation
        control_times = (
            np.arange(durations.shape[0])
            * self.data["control_period"]
            * self.data["sim_timestep"]
        )

        print(f"max control time = {np.max(durations)}")
        print(f"avg control time = {np.mean(durations)}")
        print(f"avg without first = {np.mean(durations[1:])}")

        plt.figure()
        plt.plot(control_times, durations)
        plt.grid()
        plt.xlabel("Time (s)")
        plt.ylabel("Controller time (s)")
        plt.title("Controller duration")

    def plot_cmd_vs_real_vel(self):
        ts = self.data["ts"]
        xs = self.data["xs"]
        cmd_vels = self.data["cmd_vels"]
        nv = int(self.data["nv"])

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        plt.figure()
        for i in range(nv):
            plt.plot(ts, xs[:, nv + i], label=f"$v_{i+1}$", color=colors[i])
        for i in range(nv):
            plt.plot(
                ts,
                cmd_vels[:, i],
                label=f"$v_{{cmd_{i+1}}}$",
                linestyle="--",
                color=colors[i],
            )
        plt.grid()
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity")
        plt.title("Actual and commanded velocity")

    def plot_dynamic_obs_dist(self):
        ts = self.data["ts"]
        dynamic_obs_distance = self.data["dynamic_obs_distance"]

        plt.figure()
        for j in range(dynamic_obs_distance.shape[1]):
            plt.plot(ts, dynamic_obs_distance[:, j], label=f"$d_{j+1}$")
        plt.grid()
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.title("Dynamic obstacle distance")

    def _plot_line_value_vs_time(self, key, legend_prefix, indices=None):
        ts = self.data["ts"]
        ys = self.data[key]

        if indices is not None:
            for idx in indices:
                plt.plot(ts, ys[:, idx], label=f"${legend_prefix}_{idx+1}$")
        elif len(ys.shape) > 1:
            for idx in range(ys.shape[1]):
                plt.plot(ts, ys[:, idx], label=f"${legend_prefix}_{idx+1}$")
        else:
            plt.plot(ts, ys)

    # TODO rewrite more functions in terms of this (probably eliminate a lot
    # of them)
    def plot_value_vs_time(
        self, key, indices=None, legend_prefix=None, ylabel=None, title=None
    ):
        """Plot the value stored in `key` vs. time."""
        if key not in self.data:
            print(f"Key {key} not found, skipping plot.")
            return

        if legend_prefix is None:
            legend_prefix = key
        if ylabel is None:
            ylabel = key
        if title is None:
            title = f"{key} vs time"

        fig = plt.figure()

        self._plot_line_value_vs_time(key, legend_prefix, indices=indices)

        plt.grid()
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.title(title)

        ax = plt.gca()
        return ax
