import numpy as np
import matplotlib.pyplot as plt

from tray_balance_sim.recording import DATA_DRIVE_PATH
import tray_balance_sim.util as util
from liegroups import SO3

import IPython

DATA_PATHS = [
    DATA_DRIVE_PATH / path
    for path in [
        "single-object/height-1.00/h1.0_goal2_all_2021-09-08_23-35-55.npz",
        "dynamic_obs_photo_op_2021-09-13_20-19-18.npz",
    ]
]

STATIC_OBS_DATA_PATH = DATA_DRIVE_PATH / "static_obs_2021-09-13_02-25-11.npz"
# FIG_PATH = "obj_error.pdf"


def plot_object_error(ts, r_ew_ws, Q_wes, r_ow_ws, Q_wos, tf=None):
    data_length = ts.shape[0]
    if tf is not None:
        for i in range(ts.shape[0]):
            if ts[i] > tf:
                data_length = i
                break

    r_oe_es = np.zeros_like(r_ow_ws)
    for i in range(r_oe_es.shape[0]):
        C_ew = SO3.from_quaternion(Q_wes[i, :], ordering="xyzw").inv()
        r_oe_es[i, :] = C_ew.dot(r_ow_ws[i, :] - r_ew_ws[i, :])
    r_oe_e_err = r_oe_es - r_oe_es[0, :]
    r_err_norm = np.linalg.norm(r_oe_e_err, axis=1)
    print(f"max position error = {np.max(r_err_norm[:data_length])}")

    plt.figure()
    # plt.plot(ts, r_oe_e_err[:, 0], label="$x$")
    # plt.plot(ts, r_oe_e_err[:, 1], label="$y$")
    # plt.plot(ts, r_oe_e_err[:, 2], label="$z$")
    plt.plot(ts, r_err_norm, label="Position error")

    # the rotation between EE and tray should be constant throughout the
    # tracjectory, so there error is the deviation from the starting
    # orientation
    Q_oe0 = util.quat_multiply(util.quat_inverse(Q_wos[0, :]), Q_wes[0, :])
    Q_eo_err = np.zeros(Q_wos.shape[0])
    for i in range(Q_eo_err.shape[0]):
        Q_eo = util.quat_multiply(util.quat_inverse(Q_wes[i, :]), Q_wos[i, :])
        Q_eo_err[i] = util.quat_error(util.quat_multiply(Q_oe0, Q_eo))
    # print(f"max angle error = {np.max(Q_eo_err)}")

    # plt.plot(ts, Q_eo_err[:, 0], label="$Q_x$")
    # plt.plot(ts, Q_eo_err[:, 1], label="$Q_y$")
    # plt.plot(ts, Q_eo_err[:, 2], label="$Q_z$")
    plt.plot(ts, Q_eo_err, label="Angle error")
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Error")


def main():
    for path in DATA_PATHS:
        print(path.name)
        with np.load(path) as data:
            ts = data["ts"]
            r_ew_ws = data["r_ew_ws"]
            Q_wes = data["Q_wes"]
            r_ow_ws = data["r_ow_ws"]
            Q_wos = data["Q_wos"]

        for i in range(r_ow_ws.shape[0]):
            plot_object_error(ts, r_ew_ws, Q_wes, r_ow_ws[i, :, :], Q_wos[i, :, :], tf=4)
        plt.show()

    print(STATIC_OBS_DATA_PATH.name)
    with np.load(STATIC_OBS_DATA_PATH) as data:
        ts = data["ts"]
        r_ew_ws = data["r_ew_ws"]
        Q_wes = data["Q_wes"]
        r_ow_ws = data["r_ow_ws"]
        Q_wos = data["Q_wos"]

    for i in range(r_ow_ws.shape[0]):
        plot_object_error(ts, r_ew_ws, Q_wes, r_ow_ws[i, :, :], Q_wos[i, :, :])
    plt.show()

    return

    fig = plt.figure(figsize=(3.25, 1.8))
    plt.rcParams.update({"font.size": 8, "text.usetex": True, "legend.fontsize": 8})

    # ax = plt.gca()
    plt.grid()
    r_te_e_err = r_te_es[0, :] - r_te_es
    # plt.plot(ts, r_te_e_err[:, 0], label="$x$")
    # plt.plot(ts, r_te_e_err[:, 1], label="$y$")
    # plt.plot(ts, r_te_e_err[:, 2], label="$z$")
    plt.plot(ts, np.linalg.norm(r_te_e_err, axis=1), label=r"$\mathrm{Position}$")

    # TODO need a good error metric, like axis angle
    # the rotation between EE and tray should be constant throughout the
    # trajectory, so there error is the deviation from the starting
    # orientation
    Q_et0 = Q_ets[0, :]
    Q_te0 = quat_inverse(Q_et0)
    angle_err = np.zeros(Q_ets.shape[0])
    for i in range(angle_err.shape[0]):
        angle_err[i] = quat_error(quat_multiply(Q_te0, Q_ets[i, :]))

    plt.plot(ts, angle_err, label=r"$\mathrm{Angle}$")
    plt.legend()
    plt.xlabel(r"$\mathrm{Time\ (s)}$")
    plt.ylabel(r"$\mathrm{Error}$")

    # ax.set_xticks([0, 40, 80, 120, 160])

    fig.tight_layout(pad=0.1)
    # fig.savefig(FIG_PATH)
    # print("Saved figure to {}".format(FIG_PATH))

    plt.show()


if __name__ == "__main__":
    main()
