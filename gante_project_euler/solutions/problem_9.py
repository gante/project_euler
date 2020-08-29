""" Solution for Project Euler's problem #9 """

import os
import time
from datetime import timedelta

import numpy as np


def get_solution():
    """ Solves the problem and returns the answer.
    """
    # a + b + c = 1000  -> c = 1000 - a - b [1]
    # a^2 + b^2 = c^2   -> (1000 - a - b)^2 = a^2 + b^2
    #                   -> 1M - 2k a - 2k b + 2 ab + a^2 + b^2 = a^2 + b^2
    #                   -> 500k + ab = 1000 a + 1000 b
    for b in np.arange(2, 500).astype(int):
        for a in np.arange(1, b).astype(int):
            if 1000*a + 1000*b == 500000 + a*b:
                c = np.sqrt(np.sum(np.power([a, b], 2))).astype(int)
                assert a + b + c == 1000
                return a * b * c


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    start = time.time()
    solution = get_solution()
    end = time.time()
    print("Solution: {}".format(solution))
    print("Elapsed time (w/compile time):  {} (HH:MM:SS.us)".format(timedelta(seconds=end-start)))
    # The the code is compiled the first time it runs. This second run uses the cached compilation.
    start = time.time()
    _ = get_solution()
    end = time.time()
    print("Elapsed time (wo/compile time): {} (HH:MM:SS.us)".format(timedelta(seconds=end-start)))
