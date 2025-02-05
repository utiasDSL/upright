#!/bin/sh

CONFIG=/home/adam/phd/code/projects/upright/catkin_ws/src/upright/upright_robust/config/demos/sim.yaml

set -x  # echo commands

# ./planning_sim_loop.py --config "$CONFIG" --height 60 --com robust --log
./planning_sim_loop.py --config "$CONFIG" --height 50 --com robust --log
./planning_sim_loop.py --config "$CONFIG" --height 40 --com robust --log
./planning_sim_loop.py --config "$CONFIG" --height 30 --com robust --log

# ./planning_sim_loop.py --config "$CONFIG" --height 60 --com top --log
# ./planning_sim_loop.py --config "$CONFIG" --height 50 --com top --log
# ./planning_sim_loop.py --config "$CONFIG" --height 40 --com top --log
# ./planning_sim_loop.py --config "$CONFIG" --height 30 --com top --log
#
# ./planning_sim_loop.py --config "$CONFIG" --height 60 --com center --log
# ./planning_sim_loop.py --config "$CONFIG" --height 50 --com center --log
# ./planning_sim_loop.py --config "$CONFIG" --height 40 --com center --log
# ./planning_sim_loop.py --config "$CONFIG" --height 30 --com center --log
