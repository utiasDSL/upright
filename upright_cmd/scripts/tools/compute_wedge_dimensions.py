#!/usr/bin/env python3
import numpy as np


DESIRED_ANGLE_DEG = 15
DESIRED_X_LENGTH = 0.15
DESIRED_Y_LENGTH = 0.15


def main():
    z_length = DESIRED_X_LENGTH * np.tan(np.deg2rad(DESIRED_ANGLE_DEG))
    print(f"side lengths = {[DESIRED_X_LENGTH, DESIRED_Y_LENGTH, z_length]}")

main()
