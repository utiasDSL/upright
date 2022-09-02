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

    d = 0.1
    x = np.linspace(0, 2, 100)
    y1 = x - d
    y2 = np.sqrt(x**2 + d**2) - 2*d

    plt.plot(x, y1, label="y1")
    plt.plot(x, y2, label="y2")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
