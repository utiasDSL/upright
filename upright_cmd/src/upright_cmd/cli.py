import argparse


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

# def parse_cli_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", required=True, help="Path to configuration file.")
#     parser.add_argument(
#         "--log",
#         nargs="?",
#         default=None,
#         const="",
#         help="Log data. Optionally specify prefix for log directoy.",
#     )
#     parser.add_argument(
#         "--video",
#         nargs="?",
#         default=None,
#         const="",
#         help="Record video. Optionally specify prefix for video directoy.",
#     )
#     return parser.parse_args()
