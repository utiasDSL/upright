"""Open or close real Panda gripper."""
import time
import sys

from perls2.robots.real_panda_interface import RealPandaInterface

import tray_balance_constraints as core


def main():
    try:
        cmd = sys.argv[1].lower()
    except IndexError:
        print("Command required: `open` or `close`")
        return 1

    config = core.parsing.load_config("../config/simple_joint_vel_demo.yaml")
    robot = RealPandaInterface(config, controlType="JointVelocity")

    # wait a bit to give user time to do anything necessary, like place an
    # object to be gripped
    time.sleep(5)

    if "close".startswith(cmd):
        robot.close_gripper()
    elif "open".startswith(cmd):
        robot.open_gripper()
    else:
        print(f"Unrecognized command: {cmd}. Exiting.")

    robot.disconnect()


if __name__ == "__main__":
    main()
