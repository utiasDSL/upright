import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = "../balance_data_2021-09-02_20-02-07.npz"
FIG_PATH = "ee_pose.pdf"


def main():
    with np.load(DATA_PATH) as data:
        ts = data["ts"]
        r_ew_ws = data["r_ew_ws"]
        r_ew_wds = data["r_ew_wds"]

    fig = plt.figure(figsize=(3.25, 1.8))
    plt.rcParams.update({"font.size": 8, "text.usetex": True, "legend.fontsize": 8})

    ax = plt.gca()
    plt.grid()
    plt.plot(ts, r_ew_wds[:, 0], label="$x_d$", color="r", linestyle="--")
    plt.plot(ts, r_ew_wds[:, 1], label="$y_d$", color="g", linestyle="--")
    plt.plot(ts, r_ew_wds[:, 2], label="$z_d$", color="b", linestyle="--")
    plt.plot(ts, r_ew_ws[:, 0], label="$x$", color="r")
    plt.plot(ts, r_ew_ws[:, 1], label="$y$", color="g")
    plt.plot(ts, r_ew_ws[:, 2], label="$z$", color="b")
    plt.legend()
    plt.xlabel(r"$\mathrm{Time\ (s)}$")
    plt.ylabel(r"$\mathrm{Position\ (m)}$")

    # ax.set_xticks([0, 40, 80, 120, 160])

    fig.tight_layout(pad=0.1)
    fig.savefig(FIG_PATH)
    print("Saved figure to {}".format(FIG_PATH))

    plt.show()


if __name__ == "__main__":
    main()
