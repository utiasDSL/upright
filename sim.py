import pybullet as pyb
import time
import math
from datetime import datetime

# import pybullet_data

import IPython


SIM_DT = 0.001


def main():
    clid = pyb.connect(pyb.GUI)

    ur10_base_position = [0, 0, 0]
    ur10_base_orientation = [0, 0, 0, 1]
    ur10 = pyb.loadURDF(
        "assets/urdf/ur10.urdf", ur10_base_position, ur10_base_orientation
    )
    # pyb.setPhysicsEngineParameter(enableConeFriction=0)

    pyb.setTimeStep(SIM_DT)

    t = 0

    # simulation loop
    while True:
        pyb.stepSimulation()
        t += SIM_DT


if __name__ == "__main__":
    main()
