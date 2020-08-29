""" Module that contains prime-related operations """


import numpy as np
import jax.numpy as jnp
import jax.lax as lax


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


def factorise(number, primes):
    """
    [JAX] Factorises a given number `number`, returning an array with how many times a given prime
    factorises `number` To parallelise the process, an array of primes is passed.

    :param number: single positive integer
    :param primes: array of positive integers, containing prime numbers
    :returns: array of integers with the same dimensions as `primes`, containing how many times a
        given prime factorises `number`
    """
    number = number.astype(dtype=jnp.int32)
    factors = jnp.zeros(shape=(primes.shape[0]))
    used_factors = is_factor(number, primes)

    def _cond_fn(fn_inputs):
        """ Continue while loop while there are any factors to be used """
        _, used_factors, _ = fn_inputs
        return jnp.any(used_factors)

    def _loop_fn(fn_inputs):
        """ On each loop: counts used factors, divides `number` by those factors, and then
        computes the next round of factors to use.
        """
        factors, used_factors, number = fn_inputs
        factors += used_factors
        number /= jnp.prod(jnp.where(used_factors, primes, 1))
        used_factors = is_factor(number, primes)
        return [factors, used_factors, number.astype(dtype=jnp.int32)]

    factors, _, _ = lax.while_loop(_cond_fn, _loop_fn, [factors, used_factors, number])
    return factors


def get_all_primes(*, max_prime=None, n_primes=None):
    """
    [NP] Returns a list of all primes up to the specified limit (including it). The limit can
    either be the maximum prime returned, or the number of prime numbers returned. Note that 1
    is not a prime.

    Implements the Sieve of Eratosthenes.

    :param max_prime: a positive integer (bigger than 2)
    :param n_primes: a positive integer (bigger than 1)
    """
    assert max_prime is not None or n_primes is not None, "One named input must be given"
    assert max_prime is None or max_prime > 2, "`max_prime` must be bigger than 2"
    assert n_primes is None or n_primes > 1, "`n_primes` must be bigger than 1"

    max_prime = max_prime or 2**20
    n_primes = n_primes or max_prime
    all_integers = np.arange(max_prime+1)
    valid_primes = np.ones(all_integers.shape[0], dtype=bool)

    # stops searching at ceil(sqrt(max_prime+1)) as all remaining candidates will be primes
    for integer in range(2, np.ceil(np.sqrt(max_prime+1)).astype(int)):
        if not valid_primes[integer]:
            continue
        prime = integer
        prime_candidates = all_integers[valid_primes]
        prime_candidates = prime_candidates[prime_candidates > prime]
        # All candidates whose division is equal to the floor division by the current prime are
        # its multiple, and thus not a prime
        divisions = prime_candidates / prime
        floor_divisions = np.floor(divisions)
        multiples = prime_candidates[np.isclose(divisions, floor_divisions, rtol=0.0)]
        valid_primes[multiples] = False

    # Discards 0 and 1
    all_primes = all_integers[valid_primes][2:]
    if len(all_primes) > n_primes:
        all_primes = all_primes[:n_primes]
    return all_primes
