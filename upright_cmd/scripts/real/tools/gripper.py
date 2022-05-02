"""Open or close real Panda gripper."""
import time

from perls2.robots.real_panda_interface import RealPandaInterface

import tray_balance_constraints as core
import upright_cmd as cmd


def main():
    parser = cmd.cli.basic_arg_parser()
    parser.add_argument("command", choices=["open", "close"])
    parser.add_argument("--wait", type=float, default=0)
    args = parser.parse_args()

    if args.wait < 0:
        raise ValueError("Wait time cannot be negative.")

    config = core.parsing.load_config(args.config)
    robot = RealPandaInterface(config["perls2"], controlType="JointVelocity")

    try:
        # wait a bit to give user time to do anything necessary, like place an
        # object to be gripped
        time.sleep(args.wait)

        if args.command == "close":
            robot.close_gripper()
        else:
            robot.open_gripper()
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
