""" Solution for Project Euler's problem #2 """

import time
from datetime import timedelta

import jax.numpy as jnp
from jax import jit

from gante_project_euler.math.sequences import fibonacci_up_to
from gante_project_euler.math.prime import is_multiple


@jit
def compute_solution(fibonacci_numbers):
    """ Auxiliary function to compute the solution to the problem (sum of even fibonacci numbers
    that do not exceed 4 million)

    NOTE: the original plan was to get an array with the even fibonacci numbers. However,
    the output of `fibonacci_numbers[is_even]` has unknown length at compile time, and thus
    JAX returns an error.
    There are ways to work around it, see https://github.com/google/jax/issues/2765
    """
    is_even = is_multiple(numbers=fibonacci_numbers, base=2)
    return jnp.sum(fibonacci_numbers * is_even)


def get_solution():
    """ Solves the problem and returns the answer.
    """
    fibonacci_up_to_4mil = jnp.asarray(fibonacci_up_to(value=4000000), dtype=jnp.int32)
    return compute_solution(fibonacci_up_to_4mil)


if __name__ == "__main__":
    start = time.time()
    solution = get_solution()
    end = time.time()
    print("Solution: {}".format(solution))
    print("Elapsed time: {} (HH:MM:SS.us)".format(timedelta(seconds=end-start)))
