""" Solution for Project Euler's problem #4 """

import os
import time
from datetime import timedelta

import jax.numpy as jnp
from jax import jit

from gante_project_euler.math.misc import is_palindrome


@jit
def get_sorted_multiplication_pairs():
    """ Returns all multiplication pairs required to solve this problem (all combinations of
    multiplication between two three-digit numbers). The result is sorted
    """
    array_1 = jnp.arange(100, 1000)
    array_2 = jnp.arange(100, 1000)
    all_products = jnp.outer(array_1, array_2)
    # At this point, there is some redundancy: all_products[i, j] == all_products[j, i].
    # The redundant numbers could be removed if we take the upper triangular, but it actually
    # seems to slows things down. Sorting at the tried scales seems to be faster in CPU (numpy)
    # as well.
    sorted_products = jnp.sort(all_products.flatten())
    return sorted_products


def get_solution():
    """ Solves the problem and returns the answer.
    """
    sorted_products = get_sorted_multiplication_pairs()
    reverse_products = sorted_products[::-1]
    for number in reverse_products:
        if is_palindrome(number):
            return number
    return None


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    start = time.time()
    solution = get_solution()
    end = time.time()
    print("Solution: {}".format(solution))
    print("Elapsed time: {} (HH:MM:SS.us)".format(timedelta(seconds=end-start)))
