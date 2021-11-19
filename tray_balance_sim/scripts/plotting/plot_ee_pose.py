import numpy as np
import matplotlib.pyplot as plt

from tray_balance_sim.recording import DATA_DRIVE_PATH

import IPython

DATA_PATH = DATA_DRIVE_PATH / "balance_data_2021-09-08_21-27-10.npz"
FIG_PATH = "ee_pose.pdf"


def main():
    with np.load(DATA_PATH) as data:
        ts = data["ts"]
        r_ew_ws = data["r_ew_ws"]
        r_ew_wds = data["r_ew_wds"]

    fig = plt.figure(figsize=(3.25, 1.8))
    plt.rcParams.update({"font.size": 8, "text.usetex": True, "legend.fontsize": 8})

    r_ew_w_err = r_ew_wds - r_ew_ws
    err_norm = np.linalg.norm(r_ew_w_err, axis=1)

    for i in range(err_norm.shape[0]):
        if err_norm[i] <= 0.01:
            print(f"convergence time = {ts[i]} s")
            break

    # ax = plt.gca()
    plt.grid()
    # line_xd, = plt.plot(ts, r_ew_wds[:, 0], label="$x_d$", linestyle="--")
    # line_yd, = plt.plot(ts, r_ew_wds[:, 1], label="$y_d$", linestyle="--")
    # line_zd, = plt.plot(ts, r_ew_wds[:, 2], label="$z_d$", linestyle="--")
    # plt.plot(ts, r_ew_ws[:, 0], label="$x$", color=line_xd.get_color())
    # plt.plot(ts, r_ew_ws[:, 1], label="$y$", color=line_yd.get_color())
    # plt.plot(ts, r_ew_ws[:, 2], label="$z$", color=line_zd.get_color())
    # plt.legend()
    plt.plot(ts, err_norm)
    plt.xlabel(r"$\mathrm{Time\ (s)}$")
    plt.ylabel(r"$\mathrm{Position\ (m)}$")

    # ax.set_xticks([0, 40, 80, 120, 160])

    fig.tight_layout(pad=0.1)
    fig.savefig(FIG_PATH)
    print("Saved figure to {}".format(FIG_PATH))

    plt.show()


if __name__ == "__main__":
    main()
