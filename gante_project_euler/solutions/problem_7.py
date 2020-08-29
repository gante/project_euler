""" Solution for Project Euler's problem #7 """

import os
import time
from datetime import timedelta

# import jax.numpy as jnp
# from jax import jit

from gante_project_euler.math.prime import get_all_primes


def get_solution():
    """ Solves the problem and returns the answer.
    """
    return get_all_primes(n_primes=10001)[-1]


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
