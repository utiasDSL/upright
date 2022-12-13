import argparse
import glob
from pathlib import Path


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
    dir_path = Path(args.directory)

    if args.config_name is not None:
        config_path = dir_path / args.config_name
    else:
        config_files = glob.glob(dir_path.as_posix() + "/*.yaml")
        if len(config_files) == 0:
            raise FileNotFoundError(
                "Error: could not find a config file in the specified directory."
            )
        if len(config_files) > 1:
            raise FileNotFoundError(
                "Error: multiple possible config files in the specified directory. Please specify the name using the `--config_name` option."
            )
        config_path = config_files[0]

    if args.bag_name is not None:
        bag_path = dir_path / args.bag_name
    else:
        bag_files = glob.glob(dir_path.as_posix() + "/*.bag")
        if len(bag_files) == 0:
            raise FileNotFoundError(
                "Error: could not find a bag file in the specified directory."
            )
        if len(config_files) > 1:
            print(
                "Error: multiple bag files in the specified directory. Please specify the name using the `--bag_name` option."
            )
        bag_path = bag_files[0]
    return config_path, bag_path
