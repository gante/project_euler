""" Solution for Project Euler's problem #3 """

import time
from datetime import timedelta

import jax.numpy as jnp
from jax import jit

# ***************************************************************************************
# Critical for this problem (and for any 64-bit problems)
from jax.config import config
config.update("jax_enable_x64", True)
# ***************************************************************************************

from gante_project_euler.math.prime import get_all_primes, is_factor


NUMBER = 600851475143


@jit
def compute_solution(number, primes_list):
    """ Auxiliary function to compute the solution to the problem (largest prime factor of number).
    """
    number_factors = is_factor(number=number, primes=primes_list)
    return jnp.max(primes_list * number_factors)


def get_solution():
    """ Solves the problem and returns the answer.

    NOTE: At the time of writing, JAX does not enable 64 bit operations by default, required to
    solve this problem. Check the configuration at the top of this script.
    """
    # 10k should be enough. The correct approach would be to get primes up to sqrt(NUMBER).
    max_prime = 10000
    primes_list = jnp.asarray(get_all_primes(limit=max_prime), dtype=jnp.int32)
    return compute_solution(NUMBER, primes_list)


if __name__ == "__main__":
    start = time.time()
    solution = get_solution()
    end = time.time()
    print("Solution: {}".format(solution))
    print("Elapsed time: {} (HH:MM:SS.us)".format(timedelta(seconds=end-start)))
