import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb
from pathlib import Path
from liegroups import SO3
import yaml

import tray_balance_constraints as core

import IPython


class DataLogger:
    """Log data for later saving and viewing."""
    def __init__(self, ctrl_config, sim_config):
        self.ctrl_config = ctrl_config
        self.sim_config = sim_config
        self.data = {}

    def add(self, key, value):
        """Add a single value named `key`."""
        if key in self.data:
            raise ValueError(f"Key {key} already in the data log.")
        self.data[key] = value

    def append(self, key, value):
        """Append one of a list of values to the list named `key`."""
        # copy to an array (also copies if value is already an array, which is
        # what we want)
        a = np.array(value)
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
        ctrl_conf_path = dir_path / "ctrl.yaml"
        sim_conf_path = dir_path / "sim.yaml"

        # save the recorded data
        np.savez_compressed(data_path, **self.data)

        # save the configuration used for this run
        with open(ctrl_conf_path, "w") as f:
            yaml.dump(self.ctrl_config, stream=f, default_flow_style=False)
        with open(sim_conf_path, "w") as f:
            yaml.dump(self.sim_config, stream=f, default_flow_style=False)

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

    def plot_balancing_constraints(self):
        ts = self.data["ts"]
        balancing_cons = self.data["ineq_cons"]

        plt.figure()
        for j in range(balancing_cons.shape[1]):
            plt.plot(ts, balancing_cons[:, j], label=f"$g_{j+1}$")
        plt.grid()
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.title("Balancing Inequality constraints")

    def plot_commands(self):
        ts = self.data["ts"]
        us = self.data["us"]

        plt.figure()
        for j in range(us.shape[1]):
            plt.plot(ts, us[:, j], label=f"$u_{j+1}$")
        plt.grid()
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Commanded joint acceleration")
        plt.title("Acceleration commands")

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

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        plt.figure()
        for j in range(9):
            plt.plot(ts, xs[:, 9 + j], label=f"$v_{j+1}$", color=colors[j])
        for j in range(9):
            plt.plot(
                ts,
                cmd_vels[:, j],
                label=f"$v_{{cmd_{j+1}}}$",
                linestyle="--",
                color=colors[j],
            )
        plt.grid()
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity")
        plt.title("Actual and commanded velocity")

    def plot_joint_config(self):
        ts = self.data["ts"]
        xs = self.data["xs"]

        plt.figure()
        for j in range(9):
            plt.plot(ts, xs[:, j], label=f"$q_{j+1}$")
        plt.grid()
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Joint position")
        plt.title("Joint configuration")

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


# class Recorder:
#     def __init__(
#         self,
#         dt,
#         duration,
#         period,
#         ns,
#         ni,
#         n_objects,
#         control_period=1,
#         n_balance_con=0,
#         n_dynamic_obs=0,
#         n_collision_pair=0,
#     ):
#         self.period = period
#         self.dt = dt
#         self.num_records = int(duration / (dt * period))
#
#         def zeros(x):
#             return np.zeros((self.num_records, x))
#
#         self.ts = period * dt * np.arange(self.num_records)
#         self.us = zeros(ni)
#         self.xs = zeros(ns)
#         self.xs_noisy = zeros(ns)
#         self.r_ew_ws = zeros(3)
#         self.r_ew_wds = zeros(3)
#         self.Q_wes = zeros(4)
#         self.Q_weds = zeros(4)
#         self.v_ew_ws = zeros(3)
#         self.ω_ew_ws = zeros(3)
#
#         self.r_ow_ws = np.zeros((n_objects, self.num_records, 3))
#         self.Q_wos = np.zeros((n_objects, self.num_records, 4))
#
#         # inequality constraints for balancing
#         self.ineq_cons = zeros(n_balance_con)
#
#         # TODO this may need more thinking if I add more collision pairs with
#         # the dynamic obstacle
#         # NOTE be careful in that these are not necessarily raw distance
#         # values, depending on how the constraint is formulated
#         self.dynamic_obs_distance = zeros(n_dynamic_obs)
#         self.collision_pair_distance = zeros(n_collision_pair)
#
#         # controller runs at a different frequency than the other recording
#         self.control_period = control_period
#         num_control_records = int(duration / (dt * control_period)) + 1
#         self.control_durations = np.zeros(num_control_records)
#
#     def save(self, path):
#         np.savez_compressed(
#             path,
#             ts=self.ts,
#             us=self.us,
#             xs=self.xs,
#             xs_noisy=self.xs_noisy,
#             r_ew_ws=self.r_ew_ws,
#             r_ew_wds=self.r_ew_wds,
#             Q_wes=self.Q_wes,
#             Q_weds=self.Q_weds,
#             v_ew_ws=self.v_ew_ws,
#             ω_ew_ws=self.ω_ew_ws,
#             r_ow_ws=self.r_ow_ws,
#             Q_wos=self.Q_wos,
#             ineq_cons=self.ineq_cons,
#             dynamic_obs_distance=self.dynamic_obs_distance,
#             collision_pair_distance=self.collision_pair_distance,
#             control_durations=self.control_durations,
#             control_period=self.control_period,
#         )
#         print(f"Saved data to {path}.")
#
#     def now_is_the_time(self, sim_index):
#         return sim_index % self.period == 0
#
#     def record_index(self, sim_index):
#         return sim_index // self.period
#
#     def _slice_records(self, last_sim_index):
#         """Utility common to all plots.
#
#         last_sim_index is the last simulation step before the sim ended
#         last_record_index is thus the last index of data we should plot (the rest is
#         zeros)
#         """
#         last_record_index = self.record_index(last_sim_index)
#         s = slice(0, last_record_index + 1)
#         ts = self.ts[s]
#         return s, ts
#
#     def plot_ee_position(self, last_sim_index):
#         s, ts = self._slice_records(last_sim_index)
#
#         plt.figure()
#         plt.plot(ts, self.r_ew_wds[s, 0], label="$x_d$", color="r", linestyle="--")
#         plt.plot(ts, self.r_ew_wds[s, 1], label="$y_d$", color="g", linestyle="--")
#         plt.plot(ts, self.r_ew_wds[s, 2], label="$z_d$", color="b", linestyle="--")
#         plt.plot(ts, self.r_ew_ws[s, 0], label="$x$", color="r")
#         plt.plot(ts, self.r_ew_ws[s, 1], label="$y$", color="g")
#         plt.plot(ts, self.r_ew_ws[s, 2], label="$z$", color="b")
#         plt.grid()
#         plt.legend()
#         plt.xlabel("Time (s)")
#         plt.ylabel("Position")
#         plt.title("End effector position")
#
#     def plot_ee_orientation(self, last_sim_index):
#         s, ts = self._slice_records(last_sim_index)
#
#         plt.figure()
#         plt.plot(ts, self.Q_wes[s, 0], label="$Q_{x}$", color="r")
#         plt.plot(ts, self.Q_wes[s, 1], label="$Q_{y}$", color="g")
#         plt.plot(ts, self.Q_wes[s, 2], label="$Q_{z}$", color="b")
#         plt.plot(ts, self.Q_weds[s, 0], label="$Q_{d,x}$", color="r", linestyle="--")
#         plt.plot(ts, self.Q_weds[s, 1], label="$Q_{d,y}$", color="g", linestyle="--")
#         plt.plot(ts, self.Q_weds[s, 2], label="$Q_{d,z}$", color="b", linestyle="--")
#         plt.grid()
#         plt.legend()
#         plt.xlabel("Time (s)")
#         plt.ylabel("Orientation")
#         plt.title("End effector orientation")
#
#     def plot_ee_velocity(self, last_sim_index):
#         s, ts = self._slice_records(last_sim_index)
#
#         plt.figure()
#         plt.plot(ts, self.v_ew_ws[s, 0], label="$v_x$")
#         plt.plot(ts, self.v_ew_ws[s, 1], label="$v_y$")
#         plt.plot(ts, self.v_ew_ws[s, 2], label="$v_z$")
#         plt.plot(ts, self.ω_ew_ws[s, 0], label="$ω_x$")
#         plt.plot(ts, self.ω_ew_ws[s, 1], label="$ω_y$")
#         plt.plot(ts, self.ω_ew_ws[s, 2], label="$ω_z$")
#         plt.grid()
#         plt.legend()
#         plt.xlabel("Time (s)")
#         plt.ylabel("Velocity")
#         plt.title("End effector velocity")
#
#     def plot_object_error(self, last_sim_index, obj_index):
#         s, ts = self._slice_records(last_sim_index)
#
#         r_ow_ws = self.r_ow_ws[obj_index, s, :]
#         Q_wos = self.Q_wos[obj_index, s, :]
#
#         r_oe_es = np.zeros_like(r_ow_ws)
#         for i in range(r_oe_es.shape[0]):
#             C_ew = SO3.from_quaternion(self.Q_wes[i, :], ordering="xyzw").inv()
#             r_oe_es[i, :] = C_ew.dot(r_ow_ws[i, :] - self.r_ew_ws[i, :])
#         r_oe_e_err = r_oe_es - r_oe_es[0, :]
#
#         plt.figure()
#         plt.plot(ts, r_oe_e_err[:, 0], label="$x$")
#         plt.plot(ts, r_oe_e_err[:, 1], label="$y$")
#         plt.plot(ts, r_oe_e_err[:, 2], label="$z$")
#         plt.plot(ts, np.linalg.norm(r_oe_e_err, axis=1), label="$||r||$")
#
#         # the rotation between EE and tray should be constant throughout the
#         # tracjectory, so there error is the deviation from the starting
#         # orientation
#         Q_ow0 = core.math.quat_inverse(Q_wos[0, :])
#         Q_oe0 = core.math.quat_multiply(Q_ow0, self.Q_wes[0, :])
#         Q_eo_err = np.zeros_like(Q_wos)
#         angles = np.zeros(Q_wos.shape[0])
#         for i in range(Q_eo_err.shape[0]):
#             try:
#                 Q_ew = core.math.quat_inverse(self.Q_wes[i, :])
#                 Q_eo = core.math.quat_multiply(Q_ew, Q_wos[i, :])
#                 Q_eo_err[i, :] = core.math.quat_multiply(Q_oe0, Q_eo)
#                 angles[i] = core.math.quat_error(Q_eo_err[i, :])
#             except ValueError as e:
#                 IPython.embed()
#
#         plt.plot(ts, Q_eo_err[:, 0], label="$Q_x$")
#         plt.plot(ts, Q_eo_err[:, 1], label="$Q_y$")
#         plt.plot(ts, Q_eo_err[:, 2], label="$Q_z$")
#         plt.plot(ts, angles, label=r"$\theta$")
#         plt.grid()
#         plt.legend()
#         plt.xlabel("Time (s)")
#         plt.ylabel("Error")
#         plt.title(f"Object {obj_index} error")
#
#     def plot_r_oe_e_error(self, last_sim_index):
#         s, ts = self._slice_records(last_sim_index)
#
#         plt.figure()
#         r_oe_e_err = self.r_oe_es[0, :] - self.r_oe_es[s, :]
#         plt.plot(ts, r_oe_e_err[:, 0], label="$x$")
#         plt.plot(ts, r_oe_e_err[:, 1], label="$y$")
#         plt.plot(ts, r_oe_e_err[:, 2], label="$z$")
#         plt.plot(ts, self.Q_eos[s, 0], label="$Q_x$")
#         plt.plot(ts, self.Q_eos[s, 1], label="$Q_y$")
#         plt.plot(ts, self.Q_eos[s, 2], label="$Q_z$")
#         plt.grid()
#         plt.legend()
#         plt.xlabel("Time (s)")
#         plt.ylabel("$r^{oe}_e$ error")
#         plt.title("$r^{oe}_e$ error")
#
#     def plot_r_ot_t_error(self, last_sim_index):
#         s, ts = self._slice_records(last_sim_index)
#
#         plt.figure()
#         r_ot_t_err = self.r_ot_ts[0, :] - self.r_ot_ts[s, :]
#         plt.plot(ts, r_ot_t_err[:, 0], label="$x$")
#         plt.plot(ts, r_ot_t_err[:, 1], label="$y$")
#         plt.plot(ts, r_ot_t_err[:, 2], label="$z$")
#         plt.plot(ts, self.Q_tos[s, 0], label="$Q_x$")
#         plt.plot(ts, self.Q_tos[s, 1], label="$Q_y$")
#         plt.plot(ts, self.Q_tos[s, 2], label="$Q_z$")
#         plt.grid()
#         plt.legend()
#         plt.xlabel("Time (s)")
#         plt.ylabel("$r^{ot}_t$ error")
#         plt.title("$r^{ot}_t$ error")
#
#     def plot_balancing_constraints(self, last_sim_index):
#         s, ts = self._slice_records(last_sim_index)
#
#         plt.figure()
#         for j in range(self.ineq_cons.shape[1]):
#             plt.plot(ts, self.ineq_cons[s, j], label=f"$g_{{{j+1}}}$")
#         plt.grid()
#         plt.legend()
#         plt.xlabel("Time (s)")
#         plt.ylabel("Value")
#         plt.title("Inequality constraints")
#
#     def plot_commands(self, last_sim_index):
#         s, ts = self._slice_records(last_sim_index)
#
#         plt.figure()
#         for j in range(self.us.shape[1]):
#             plt.plot(ts, self.us[s, j], label=f"$u_{j+1}$")
#         plt.grid()
#         plt.legend()
#         plt.xlabel("Time (s)")
#         plt.ylabel("Commanded joint acceleration")
#         plt.title("Acceleration commands")
#
#     def plot_control_durations(self, last_sim_index):
#         last_record_index = last_sim_index // self.control_period
#         ts = np.arange(last_record_index + 1) * self.control_period * self.dt
#
#         durations = self.control_durations[: last_record_index + 1]
#
#         print(f"max control time = {np.max(durations)}")
#         print(f"avg control time = {np.mean(durations)}")
#         print(f"avg without first = {np.mean(durations[1:])}")
#
#         plt.figure()
#         plt.plot(ts, durations)
#         plt.grid()
#         plt.xlabel("Time (s)")
#         plt.ylabel("Controller time (s)")
#         plt.title("Controller duration")
#
#     def plot_cmd_vs_real_vel(self, last_sim_index):
#         s, ts = self._slice_records(last_sim_index)
#
#         plt.figure()
#         for j in range(9):
#             plt.plot(ts, self.xs[s, 9 + j], label=f"$v_{{{j+1}}}$")
#         for j in range(9):
#             plt.plot(
#                 ts, self.cmd_vels[s, j], label=f"$v_{{cmd_{j+1}}}$", linestyle="--"
#             )
#         plt.grid()
#         plt.legend()
#         plt.xlabel("Time (s)")
#         plt.ylabel("Velocity")
#         plt.title("Actual and commanded velocity")
#
#     def plot_joint_config(self, last_sim_index):
#         s, ts = self._slice_records(last_sim_index)
#
#         plt.figure()
#         for j in range(9):
#             plt.plot(ts, self.xs[s, j], label=f"$q_{j+1}$")
#         plt.grid()
#         plt.legend()
#         plt.xlabel("Time (s)")
#         plt.ylabel("Joint position")
#         plt.title("Joint configuration")
#
#     def plot_dynamic_obs_dist(self, last_sim_index):
#         s, ts = self._slice_records(last_sim_index)
#
#         plt.figure()
#         for j in range(self.dynamic_obs_distance.shape[1]):
#             plt.plot(ts, self.dynamic_obs_distance[s, j], label=f"$d_{j+1}$")
#         plt.grid()
#         plt.legend()
#         plt.xlabel("Time (s)")
#         plt.ylabel("Distance (m)")
#         plt.title("Dynamic obstacle distance")
