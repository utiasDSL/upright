import numpy as np
import time

from upright_core.util import Rate

import IPython


def main():
    hz = 1000
    rate = Rate(1e6)

    t = 0
    dt = rate.timestep_secs

    start = time.time()
    while t < 5:
        time.sleep(0.5 * dt)
        rate.sleep()
        # print(f"clock time = {time.time() - start}")
        t += dt


if __name__ == "__main__":
    main()
