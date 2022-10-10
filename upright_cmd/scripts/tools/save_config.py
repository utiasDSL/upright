#!/usr/bin/env python3
import argparse
import yaml

import upright_core as core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration file.")
    parser.add_argument("--output", required=True, help="Path to output file.")
    args = parser.parse_args()

    # load configuration and write it out as a single yaml file
    config = core.parsing.load_config(args.config)
    with open(args.output, "w") as f:
        yaml.dump(config, stream=f, default_flow_style=False)


if __name__ == "__main__":
    main()
