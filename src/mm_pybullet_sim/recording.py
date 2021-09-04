import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import mm_pybullet_sim.util as util

import IPython

DATA_DRIVE_PATH = Path("/media/adam/Data/PhD/Data/ICRA22")


class Recorder:
    def __init__(
        self,
        dt,
        duration,
        period,
        model,
        control_period=1,
        n_balance_con=0,
        n_dynamic_obs=0,
        n_collision_pair=0,
    ):
        self.period = period
        self.dt = dt
        num_records = int(duration / (dt * period)) + 1

        def zeros(x):
            return np.zeros((num_records, x))

        self.ts = period * dt * np.arange(num_records)
        self.us = zeros(model.ni)
        self.xs = zeros(model.ns)
        self.r_ew_ws = zeros(3)
        self.r_ew_wds = zeros(3)
        self.Q_wes = zeros(4)
        self.Q_des = zeros(4)
        self.v_ew_ws = zeros(3)
        self.ω_ew_ws = zeros(3)
        self.r_tw_ws = zeros(3)

        # inequality constraints for balancing
        self.ineq_cons = zeros(n_balance_con)

        self.r_te_es = zeros(3)
        self.Q_ets = zeros(4)

        self.r_oe_es = zeros(3)
        self.Q_eos = zeros(4)

        self.r_ot_ts = zeros(3)
        self.Q_tos = zeros(4)

        # TODO this may need more thinking if I add more collision pairs with
        # the dynamic obstacle
        # NOTE be careful in that these are not necessarily raw distance
        # values, depending on how the constraint is formulated
        self.dynamic_obs_distance = zeros(n_dynamic_obs)
        self.collision_pair_distance = zeros(n_collision_pair)

        # controller runs at a different frequency than the other recording
        self.control_period = control_period
        num_control_records = int(duration / (dt * control_period)) + 1
        self.control_durations = np.zeros(num_control_records)

    def save(self, filename, use_data_drive=True):
        if use_data_drive:
            path = DATA_DRIVE_PATH / filename
        else:
            path = filename
        np.savez_compressed(
            path,
            ts=self.ts,
            us=self.us,
            r_ew_ws=self.r_ew_ws,
            r_ew_wds=self.r_ew_wds,
            Q_wes=self.Q_wes,
            Q_des=self.Q_des,
            v_ew_ws=self.v_ew_ws,
            ω_ew_ws=self.ω_ew_ws,
            r_tw_ws=self.r_tw_ws,
            ineq_cons=self.ineq_cons,
            r_te_es=self.r_te_es,
            Q_ets=self.Q_ets,
            r_oe_es=self.r_oe_es,
            Q_eos=self.Q_eos,
            r_ot_ts=self.r_ot_ts,
            Q_tos=self.Q_tos,
            dynamic_obs_distance=self.dynamic_obs_distance,
            collision_pair_distance=self.collision_pair_distance,
            control_durations=self.control_durations,
        )
        print(f"Saved data to {filename}.")

    def now_is_the_time(self, sim_index):
        return sim_index % self.period == 0

    def record_index(self, sim_index):
        return sim_index // self.period

    def _slice_records(self, last_sim_index):
        """Utility common to all plots.

        last_sim_index is the last simulation step before the sim ended
        last_record_index is thus the last index of data we should plot (the rest is
        zeros)
        """
        last_record_index = self.record_index(last_sim_index)
        s = slice(0, last_record_index)
        ts = self.ts[s]
        return s, ts

    def plot_ee_position(self, last_sim_index):
        s, ts = self._slice_records(last_sim_index)

        plt.figure()
        plt.plot(ts, self.r_ew_wds[s, 0], label="$x_d$", color="r", linestyle="--")
        plt.plot(ts, self.r_ew_wds[s, 1], label="$y_d$", color="g", linestyle="--")
        plt.plot(ts, self.r_ew_wds[s, 2], label="$z_d$", color="b", linestyle="--")
        plt.plot(ts, self.r_ew_ws[s, 0], label="$x$", color="r")
        plt.plot(ts, self.r_ew_ws[s, 1], label="$y$", color="g")
        plt.plot(ts, self.r_ew_ws[s, 2], label="$z$", color="b")
        plt.grid()
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Position")
        plt.title("End effector position")

    def plot_ee_orientation(self, last_sim_index):
        s, ts = self._slice_records(last_sim_index)

        plt.figure()
        plt.plot(ts, self.Q_des[s, 0], label="$ΔQ_x$", color="r")
        plt.plot(ts, self.Q_des[s, 1], label="$ΔQ_y$", color="g")
        plt.plot(ts, self.Q_des[s, 2], label="$ΔQ_z$", color="b")
        plt.grid()
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Orientation error")
        plt.title("End effector orientation error")

    def plot_ee_velocity(self, last_sim_index):
        s, ts = self._slice_records(last_sim_index)

        plt.figure()
        plt.plot(ts, self.v_ew_ws[s, 0], label="$v_x$")
        plt.plot(ts, self.v_ew_ws[s, 1], label="$v_y$")
        plt.plot(ts, self.v_ew_ws[s, 2], label="$v_z$")
        plt.plot(ts, self.ω_ew_ws[s, 0], label="$ω_x$")
        plt.plot(ts, self.ω_ew_ws[s, 1], label="$ω_y$")
        plt.plot(ts, self.ω_ew_ws[s, 2], label="$ω_z$")
        plt.grid()
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity")
        plt.title("End effector velocity")

    def plot_r_te_e_error(self, last_sim_index):
        s, ts = self._slice_records(last_sim_index)

        plt.figure()
        r_te_e_err = self.r_te_es[0, :] - self.r_te_es[s, :]
        plt.plot(ts, r_te_e_err[:, 0], label="$x$")
        plt.plot(ts, r_te_e_err[:, 1], label="$y$")
        plt.plot(ts, r_te_e_err[:, 2], label="$z$")
        plt.plot(ts, np.linalg.norm(r_te_e_err, axis=1), label="$||r||$")

        # the rotation between EE and tray should be constant throughout the
        # tracjectory, so there error is the deviation from the starting
        # orientation
        Q_et0 = self.Q_ets[0, :]
        Q_te0 = util.quat_inverse(Q_et0)
        Q_et_err = np.zeros_like(self.Q_ets[s, :])
        for i in range(Q_et_err.shape[0]):
            Q_et_err[i, :] = util.quat_multiply(Q_te0, self.Q_ets[i, :])

        plt.plot(ts, Q_et_err[:, 0], label="$Q_x$")
        plt.plot(ts, Q_et_err[:, 1], label="$Q_y$")
        plt.plot(ts, Q_et_err[:, 2], label="$Q_z$")
        plt.grid()
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("$r^{te}_e$ error")
        plt.title("$r^{te}_e$ error")

    def plot_r_oe_e_error(self, last_sim_index):
        s, ts = self._slice_records(last_sim_index)

        plt.figure()
        r_oe_e_err = self.r_oe_es[0, :] - self.r_oe_es[s, :]
        plt.plot(ts, r_oe_e_err[:, 0], label="$x$")
        plt.plot(ts, r_oe_e_err[:, 1], label="$y$")
        plt.plot(ts, r_oe_e_err[:, 2], label="$z$")
        plt.plot(ts, self.Q_eos[s, 0], label="$Q_x$")
        plt.plot(ts, self.Q_eos[s, 1], label="$Q_y$")
        plt.plot(ts, self.Q_eos[s, 2], label="$Q_z$")
        plt.grid()
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("$r^{oe}_e$ error")
        plt.title("$r^{oe}_e$ error")

    def plot_r_ot_t_error(self, last_sim_index):
        s, ts = self._slice_records(last_sim_index)

        plt.figure()
        r_ot_t_err = self.r_ot_ts[0, :] - self.r_ot_ts[s, :]
        plt.plot(ts, r_ot_t_err[:, 0], label="$x$")
        plt.plot(ts, r_ot_t_err[:, 1], label="$y$")
        plt.plot(ts, r_ot_t_err[:, 2], label="$z$")
        plt.plot(ts, self.Q_tos[s, 0], label="$Q_x$")
        plt.plot(ts, self.Q_tos[s, 1], label="$Q_y$")
        plt.plot(ts, self.Q_tos[s, 2], label="$Q_z$")
        plt.grid()
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("$r^{ot}_t$ error")
        plt.title("$r^{ot}_t$ error")

    def plot_balancing_constraints(self, last_sim_index):
        s, ts = self._slice_records(last_sim_index)

        plt.figure()
        for j in range(self.ineq_cons.shape[1]):
            plt.plot(ts, self.ineq_cons[s, j], label=f"$g_{j+1}$")
        plt.grid()
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.title("Inequality constraints")

    def plot_commands(self, last_sim_index):
        s, ts = self._slice_records(last_sim_index)

        plt.figure()
        for j in range(self.us.shape[1]):
            plt.plot(ts, self.us[s, j], label=f"$u_{j+1}$")
        plt.grid()
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Commanded joint acceleration")
        plt.title("Acceleration commands")

    def plot_control_durations(self):
        ts = np.arange(self.control_durations.shape[0]) * self.control_period * self.dt

        print(f"max control time = {np.max(self.control_durations)}")
        print(f"avg control time = {np.mean(self.control_durations)}")
        print(f"avg without first = {np.mean(self.control_durations[1:])}")

        plt.figure()
        plt.plot(ts, self.control_durations)
        plt.grid()
        plt.xlabel("Time (s)")
        plt.ylabel("Controller time (s)")
        plt.title("Controller duration")
