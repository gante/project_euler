""" Solution for Project Euler's problem #1 """

import os
import time
from datetime import timedelta

import jax.numpy as jnp
from jax import jit

from gante_project_euler.math.prime import is_multiple


@jit
def get_solution():
    """ Solves the problem and returns the answer.
    """
    integers = jnp.arange(1000)
    mul_3 = is_multiple(numbers=integers, base=3)
    mul_5 = is_multiple(numbers=integers, base=5)
    multiples_of_3_or_5 = mul_3 | mul_5
    return jnp.sum(integers[multiples_of_3_or_5])


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    start = time.time()
    solution = get_solution()
    end = time.time()
    print("Solution: {}".format(solution))
    print("Elapsed time: {} (HH:MM:SS.us)".format(timedelta(seconds=end-start)))
