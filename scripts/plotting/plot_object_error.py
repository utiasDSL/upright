import numpy as np
import matplotlib.pyplot as plt

from mm_pybullet_sim.recording import DATA_DRIVE_PATH
from mm_pybullet_sim.util import quat_inverse, quat_multiply, quat_error

DATA_PATH = DATA_DRIVE_PATH / "balance_data_2021-09-02_21-29-09.npz"
FIG_PATH = "obj_error.pdf"


def main():
    with np.load(DATA_PATH) as data:
        ts = data["ts"]
        r_te_es = data["r_te_es"]
        Q_ets = data["Q_ets"]

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
    fig.savefig(FIG_PATH)
    print("Saved figure to {}".format(FIG_PATH))

    plt.show()


if __name__ == "__main__":
    main()
