#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import IPython


def epsnorm_high(x, ε):
    """Always larger than ||x||"""
    return np.sqrt(x @ x + ε)

def epsnorm_low(x, ε):
    """Always smaller than ||x||"""
    return epsnorm_high(x, ε**2) - ε


def main():
    # ε = 0.001
    # for i in range(1000):
    #     x = np.random.random(3)
    #     norm = np.linalg.norm(x)
    #     low = epsnorm_low(x, ε)
    #     high = epsnorm_high(x, ε)
    #
    #     print(norm - low)
    #
    #     if norm > high or norm < low:
    #         IPython.embed()
    #         return

    d_min = 0.1
    d = np.linspace(-0.5, 2, 500)
    y1 = d - d_min
    y2 = np.sign(d) * np.sqrt(d**2 + d_min**2) - d_min

    plt.plot(d, y1, label="y1")
    plt.plot(d, y2, label="y2")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
