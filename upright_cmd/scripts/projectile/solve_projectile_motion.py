#!/usr/bin/env python3
"""Compute initial velocity for desired projectile flight between two points."""
import numpy as np
import IPython


def main():
    T = 0.75  # flight time
    g = np.array([0, 0, -9.81])  # gravity

    # desired start and end positions
    rf = np.zeros(3)
    r0 = np.array([np.sqrt(2), np.sqrt(2), 0])

    # solve for required initial velocity
    v = (rf - r0) / T - 0.5 * T * g

    print(v)


main()
