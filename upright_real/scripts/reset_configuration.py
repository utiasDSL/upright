"""Reset real Panda robot to neutral position given in config file."""
from perls2.robots.real_panda_interface import RealPandaInterface

import tray_balance_constraints as core


def main():
    config = core.parsing.load_config("../config/follow_plan_demo.yaml")
    robot = RealPandaInterface(config, controlType="JointVelocity")
    robot.reset()
    robot.disconnect()


if __name__ == "__main__":
    main()
