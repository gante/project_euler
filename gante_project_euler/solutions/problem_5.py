""" Solution for Project Euler's problem #5 """

import os
import time
from datetime import timedelta

import jax.numpy as jnp
from jax import jit, ops

from gante_project_euler.math.prime import get_all_primes, factorise


MAX_NUMBER = 20


@jit
def compute_solution(primes_list):
    """ Auxiliary function to compute the solution to the problem (smallest positive number that
    is evenly divisible by all of the numbers from 1 to 20).
    """
    all_factors = jnp.zeros(shape=(MAX_NUMBER+1, primes_list.shape[0]))
    # TODO: replace by vmap?
    for number in jnp.arange(2, MAX_NUMBER+1):
        # numpy equivalent -> all_factors[number, :] += factorise(number=number, primes=primes_list)
        all_factors = ops.index_add(
            all_factors,
            ops.index[number, :],
            factorise(number=number, primes=primes_list)
        )
    # Being evenly divisible by all those numbers is equivalent to being divisible by the largest
    # factorisation of each prime for those numbers
    max_factors = jnp.max(all_factors, axis=0).astype(dtype=jnp.int32)
    return jnp.prod(jnp.power(primes_list, max_factors))


def get_solution():
    """ Solves the problem and returns the answer.
    """
    max_prime = MAX_NUMBER
    primes_list = jnp.asarray(get_all_primes(limit=max_prime), dtype=jnp.int32)
    return compute_solution(primes_list)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    start = time.time()
    solution = get_solution()
    end = time.time()
    print("Solution: {}".format(solution))
    print("Elapsed time: {} (HH:MM:SS.us)".format(timedelta(seconds=end-start)))
