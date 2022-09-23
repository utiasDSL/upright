#!/bin/sh
LOG_DIR=/media/adam/Data/PhD/Data/upright/experiments/thing/bags/$(date +"%Y-%m-%d")/$(date +"%H-%M-%S")
mkdir -p "$LOG_DIR"

# save configuration dictionary
rosrun upright_cmd save_config.py --config "$1" --output "$LOG_DIR"/config.yaml

rosbag record -o "$LOG_DIR/bag" \
  /clock \
  --regex "/ridgeback/(.*)" \
  --regex "/ridgeback_velocity_controller/(.*)" \
  --regex "/ur10/(.*)" \
  --regex "/vicon/(.*)" \
  --regex "/mobile_manipulator_(.*)"
