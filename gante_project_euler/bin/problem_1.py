""" Solution for Project Euler's problem #1 """

import time
from datetime import timedelta

import jax.numpy as jnp
from jax import jit

from gante_project_euler.math.prime import is_multiple


@jit
def main():
    """ Solves the problem and returns the answer.
    """
    integers = jnp.arange(1000)
    mul_3 = is_multiple(integers, 3)
    mul_5 = is_multiple(integers, 5)
    multiples_of_3_or_5 = mul_3 | mul_5
    sum_of_multiples = jnp.sum(integers[multiples_of_3_or_5])
    return sum_of_multiples


if __name__ == "__main__":
    start = time.time()
    solution = main()
    end = time.time()
    print("Solution: {}".format(solution))
    print("Elapsed time: {} (HH:MM:SS.us)".format(timedelta(seconds=end-start)))
