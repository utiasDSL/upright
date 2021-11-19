import numpy as np
import matplotlib.pyplot as plt

from tray_balance_sim.recording import DATA_DRIVE_PATH

DATA_PATH = DATA_DRIVE_PATH / "balance_data_2021-09-02_21-29-09.npz"
FIG_PATH = "ineq_cons.pdf"


def main():
    with np.load(DATA_PATH) as data:
        ts = data["ts"]
        ineq_cons = data["ineq_cons"]

    fig = plt.figure(figsize=(3.25, 1.8))
    plt.rcParams.update({"font.size": 8, "text.usetex": True, "legend.fontsize": 8})

    # ax = plt.gca()
    plt.grid()
    plt.plot(ts, ineq_cons[:, 0], label=r"$\mathrm{Normal}$")
    plt.plot(ts, ineq_cons[:, 1], label=r"$\mathrm{Friction}$")
    plt.plot(ts, ineq_cons[:, 2], label=r"$\mathrm{Tipping}$")
    plt.legend()
    plt.xlabel(r"$\mathrm{Time\ (s)}$")
    plt.ylabel(r"$\mathrm{Balancing\ constraint\ value}$")

    # ax.set_xticks([0, 40, 80, 120, 160])

    fig.tight_layout(pad=0.1)
    fig.savefig(FIG_PATH)
    print("Saved figure to {}".format(FIG_PATH))

    plt.show()


if __name__ == "__main__":
    main()
