#!/usr/bin/env python3
import numpy as np

from upright_sim import geometry

import IPython


def main():
    s = 0.2
    h = geometry.equilateral_triangle_inscribed_radius(s)
    r = 2 * h

    r_tau_eqtri = geometry.equilateral_triangle_r_tau(s)
    r_tau_in = geometry.circle_r_tau(h)
    r_tau_out = geometry.circle_r_tau(r)

    print(f"triangle = {r_tau_eqtri}")
    print(f"in circ  = {r_tau_in}")
    print(f"out circ = {r_tau_out}")


if __name__ == "__main__":
    main()
