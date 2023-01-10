import numpy as np


def parse_mpc_observation_msgs(msgs, normalize_time=True):
    ts = []
    xs = []
    us = []

    for msg in msgs:
        ts.append(msg.time)
        xs.append(msg.state.value)
        us.append(msg.input.value)

    ts = np.array(ts)
    if normalize_time:
        ts -= ts[0]

    return ts, np.array(xs), np.array(us)
