#!/usr/bin/env python3
"""Plot the state x over time."""
import sys
import numpy as np

from upright_core.logging import DataPlotter


def main():
    plotter = DataPlotter.from_npz(sys.argv[1])
    plotter.plot_state()
    plotter.show()


if __name__ == "__main__":
    main()
