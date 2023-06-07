import argparse

import upright_ros_interface as rosi


def basic_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration file.")
    parser.add_argument(
        "--log",
        nargs="?",
        default=None,
        const="",
        help="Log data. Optionally specify prefix for log directoy.",
    )
    return parser


def sim_arg_parser():
    parser = basic_arg_parser()

    # sim permits video recording as well
    parser.add_argument(
        "--video",
        nargs="?",
        default=None,
        const="",
        help="Record video. Optionally specify prefix for video directoy.",
    )
    return parser


def add_bag_dir_arguments(parser):
    """Add arguments for a directory containing a bag file and config.yaml, as
    recorded by record.py."""
    parser.add_argument(
        "directory", help="Directory in which to find config and bag files."
    )
    parser.add_argument(
        "--config_name",
        help="Name of config file within the given directory.",
        required=False,
    )
    parser.add_argument(
        "--bag_name",
        help="Name of bag file within the given directory.",
        required=False,
    )
    return parser


def parse_bag_dir_args(args):
    """Parse bag and config file paths from CLI args."""
    return rosi.parsing.parse_bag_dir(
        directory=args.directory, config_name=args.config_name, bag_name=args.bag_name
    )
