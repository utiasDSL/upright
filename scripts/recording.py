import numpy as np
import matplotlib.pyplot as plt


class Recorder:
    def __init__(self, dt, duration, period, **params):
        self.period = period
        num_records = int(duration / (dt * period))

        def zeros(x):
            return np.zeros((num_records, x))

        self.ts = period * dt * np.arange(num_records)
        self.us = zeros(params["model"].ni)
        self.r_ew_ws = zeros(3)
        self.r_ew_wds = zeros(3)
        self.Q_wes = zeros(4)
        self.Q_des = zeros(4)
        self.v_ew_ws = zeros(3)
        self.ω_ew_ws = zeros(3)
        self.r_tw_ws = zeros(3)
        self.ineq_cons = zeros(params["problem"].n_balance_con)

        self.r_te_es = zeros(3)
        self.Q_ets = zeros(4)

        self.r_oe_es = zeros(3)
        self.Q_eos = zeros(4)

        self.r_ot_ts = zeros(3)
        self.Q_tos = zeros(4)

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
        plt.plot(ts, self.Q_ets[s, 0], label="$Q_x$")
        plt.plot(ts, self.Q_ets[s, 1], label="$Q_y$")
        plt.plot(ts, self.Q_ets[s, 2], label="$Q_z$")
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
