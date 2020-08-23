""" Solution for Project Euler's problem #4 """

import time
from datetime import timedelta

import jax.numpy as jnp
from jax import jit


@jit
def get_solution():
    """ Solves the problem and returns the answer.
    """
    return jnp.sum(jnp.arange(10))


if __name__ == "__main__":
    start = time.time()
    solution = get_solution()
    end = time.time()
    print("Solution: {}".format(solution))
    print("Elapsed time: {} (HH:MM:SS.us)".format(timedelta(seconds=end-start)))
