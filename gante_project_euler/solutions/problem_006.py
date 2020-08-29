""" Solution for Project Euler's problem #6 """

import os
import time
from datetime import timedelta

import jax.numpy as jnp
from jax import jit


@jit
def get_solution():
    """ Solves the problem and returns the answer.
    """
    array_up_to_100 = jnp.arange(100+1)
    sum_of_squares = jnp.sum(jnp.square(array_up_to_100))
    square_of_sum = jnp.square(jnp.sum(array_up_to_100))
    return square_of_sum - sum_of_squares


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
