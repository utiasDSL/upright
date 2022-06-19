"""Reset real Panda robot to neutral position given in config file."""
from perls2.robots.real_panda_interface import RealPandaInterface

import upright_core as core
import upright_cmd as cmd


def main():
    args = cmd.cli.basic_arg_parser().parse_args()
    config = core.parsing.load_config(args.config)
    robot = RealPandaInterface(config["perls2"], controlType="JointVelocity")
    robot.reset()
    robot.disconnect()


if __name__ == "__main__":
    main()
