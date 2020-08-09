""" Solution for Project Euler's problem #XXX """

import time
from datetime import timedelta

import jax.numpy as jnp
from jax import jit


@jit
def main():
    """ Solves the problem and returns the answer.
    """
    return jnp.sum(jnp.arange(10))


if __name__ == "__main__":
    start = time.time()
    solution = main()
    end = time.time()
    print("Solution: {}".format(solution))
    print("Elapsed time: {} (HH:MM:SS.us)".format(timedelta(seconds=end-start)))
