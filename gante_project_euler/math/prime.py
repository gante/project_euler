""" Module that contains prime-related operations """


import numpy as np
import jax.numpy as jnp


def is_multiple(numbers, base):
    """
    [JAX] For each member of `numbers`, checks if it is a multiple of `base`, returning True
    if it is the case.

    :param numbers: array of numbers
    :param base: single number
    :returns: boolean array with the same dimensions as `numbers`
    """
    division = numbers / base
    floor_division = jnp.floor(division)
    return jnp.isclose(division, floor_division, rtol=0.0)


def is_factor(number, primes):
    """
    [JAX] Gets the factors of a given `number`. To parallelise the process, an array of primes is
    passed.

    :param number: single positive integer
    :param primes: array of positive integers, containing prime numbers
    :returns: boolean array with the same dimensions as `primes`
    """
    division = number / primes
    floor_division = jnp.floor(division)
    return jnp.isclose(division, floor_division, rtol=0.0)


def get_all_primes(limit):
    """
    [NP] Returns a list of all primes up to the specified limit (including it).

    :param limit: a positive integer (bigger than 2)
    """
    assert limit > 2, "`limit` must be bigger than 2"
    all_primes = [2]
    for integer in range(3, limit+1, 2):   # skips even numbers
        is_prime = True
        for prime in all_primes:
            division = integer / prime
            floor_division = np.floor(division)
            if np.isclose(division, floor_division, rtol=0.0):
                # `prime` is a factor of `integer`, and thus `integer` is not a prime
                is_prime = False
                break
        if is_prime:
            all_primes.append(integer)
    return all_primes
