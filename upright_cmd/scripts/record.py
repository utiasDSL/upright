#!/usr/bin/env python3
import argparse
import datetime
from pathlib import Path
import subprocess
import signal
import time
import yaml

import upright_core as core

import IPython


LOG_DIR_ROOT = Path("/media/adam/Data/PhD/Data/upright/experiments/thing/bags")

ROSBAG_CMD_ROOT = ["rosbag", "record"]
# fmt: off
ROSBAG_TOPICS = [
        "/clock",
        "--regex", "/ridgeback/(.*)",
        "--regex", "/ridgeback_velocity_controller/(.*)",
        "--regex", "/ur10/(.*)",
        "--regex", "/vicon/(.*)",
        "--regex", "/projectile/(.*)",
        "--regex", "/Projectile/(.*)",
        "--regex", "/mobile_manipulator_(.*)"
]
# fmt: on


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to config file")
    parser.add_argument("--name", help="Name to be prepended to directory.")
    parser.add_argument("--notes", help="Additional information written to notes.txt inside the directory.")
    args = parser.parse_args()

    # create the log directory
    stamp = datetime.datetime.now()
    ymd = stamp.strftime("%Y-%m-%d")
    hms = stamp.strftime("%H-%M-%S")
    if args.name is not None:
        dir_name = Path(ymd) / (args.name + "_" + hms)
    else:
        dir_name = Path(ymd) / hms

    log_dir = LOG_DIR_ROOT / dir_name
    log_dir.mkdir(parents=True)

    # load configuration and write it out as a single yaml file
    config_out_path = log_dir / "config.yaml"
    config = core.parsing.load_config(args.config_path)
    with open(config_out_path, "w") as f:
        yaml.dump(config, stream=f, default_flow_style=False)

    # write any notes
    if args.notes is not None:
        notes_out_path = log_dir / "notes.txt"
        with open(notes_out_path, "w") as f:
            f.write(args.notes)

    # start the logging with rosbag
    rosbag_out_path = log_dir / "bag"
    rosbag_cmd = ROSBAG_CMD_ROOT + ["-o", rosbag_out_path] + ROSBAG_TOPICS
    proc = subprocess.Popen(rosbag_cmd)

    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)

    # IPython.embed()


if __name__ == "__main__":
    main()