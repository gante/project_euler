""" Solution for Project Euler's problem #5 """

import os
import time
from datetime import timedelta
from functools import partial

import jax.numpy as jnp
from jax import jit, vmap

from gante_project_euler.math.prime import get_all_primes, factorise


MAX_NUMBER = 20


# In this function, the first argument controls its flow (in the `vmap`, defines how many times
# the operation is applied). As such, normal `@jit` would fail. The following modification lets
# the compiler know that the 1st argument will need further compiling every time it runs.
# Doing this is still much faster than doing no compilation at all.
# NOTE: this is not required for the problem, just a fun add-on :)
# https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#python-control-flow-+-JIT
jit_with_1_controlarg = partial(jit, static_argnums=(0,))
@jit_with_1_controlarg
def compute_solution(max_number, primes_list):
    """ Auxiliary function to compute the solution to the problem (smallest positive number that
    is evenly divisible by all of the numbers from 1 to `max_number`, in this case 20).
    """
    factorise_w_primes = partial(factorise, primes=primes_list)
    # The following instruction is equivalent to getting a row of prime factors for each desired
    # number - gets a matrix with dims <max_number-1; primes_list.shape[0]>
    all_factors = vmap(factorise_w_primes)(jnp.arange(2, max_number+1))

    # Being evenly divisible by all those numbers is equivalent to being divisible by the largest
    # factorisation of each prime for those numbers
    max_factors = jnp.max(all_factors, axis=0).astype(dtype=jnp.int32)
    return jnp.prod(jnp.power(primes_list, max_factors))


def get_solution():
    """ Solves the problem and returns the answer.
    """
    primes_list = jnp.asarray(get_all_primes(max_prime=MAX_NUMBER), dtype=jnp.int32)
    return compute_solution(MAX_NUMBER, primes_list)


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
