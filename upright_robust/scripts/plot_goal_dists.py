"""Check that trajectories have converged to desired goal position."""
import argparse
import numpy as np
import matplotlib.pyplot as plt

import IPython


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "npzfile", help="NPZ file containing goal distance information."
    )
    args = parser.parse_args()

    data = np.load(args.npzfile)
    times = data["times"]
    dists = data["goal_dists"]
    n = times.shape[0]

    plt.figure()
    for i in range(n):
        plt.plot(times[i, :], dists[i, :], color="b", alpha=0.2)
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Dist [m]")
    plt.show()


main()
