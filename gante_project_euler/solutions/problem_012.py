""" Solution for Project Euler's problem #12 """

import os
import time
from datetime import timedelta
from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap

# ***************************************************************************************
# Critical for this problem (and for any 64-bit problems). When in doubt, use it
from jax.config import config
config.update("jax_enable_x64", True)
# ***************************************************************************************

from gante_project_euler.math.prime import get_all_primes, factorise
from gante_project_euler.math.sequences import triangle_up_to


# Search limits (should be enough)
MAX_INT = 2**30
MAX_PRIME = 100000


@jit
def compute_solution(primes_list, triangle_sequence):
    """ Auxiliary function to compute the solution to the problem.
    """
    factorise_w_primes = partial(factorise, primes=primes_list)
    all_factors = vmap(factorise_w_primes)(triangle_sequence)
    # number of divisors = number of possible combinations of prime factors
    # = inner product(number of states for each prime in a number)
    # e.g. 1024 has 11 states for prime=2, and 1 state for the others
    # 3072 has 11 states for prime=2 and 2 states for prime=3 -> 22 divisors
    all_factors = all_factors + 1
    n_combinations = jnp.prod(all_factors, axis=1).astype(jnp.int32)
    return n_combinations


def get_solution():
    """ Solves the problem and returns the answer.
    """
    primes_list = get_all_primes(max_prime=MAX_PRIME)
    triangle_sequence = np.asarray(triangle_up_to(value=MAX_INT), dtype=int)
    n_combinations = compute_solution(
        primes_list=jnp.asarray(primes_list, dtype=jnp.int32),
        triangle_sequence=jnp.asarray(triangle_sequence, dtype=jnp.int32)
    ).astype(int)
    # Easier to do masks outside jax :D
    return min(triangle_sequence[n_combinations > 500])


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
