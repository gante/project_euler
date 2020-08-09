""" Module that contains prime-related operations """


import jax.numpy as jnp


def is_multiple(numbers, base):
    """
    For each member of `numbers`, checks if it is a multiple of `base`, returning True
    if it is the case.

    :param numbers: array of numbers
    :param base: single number
    :returns: boolean array with the same dimensions as `numbers`
    """
    division = numbers / base
    floor_division = jnp.floor(division)
    return jnp.isclose(division, floor_division)
