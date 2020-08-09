""" Module with functions that won't fit nowhere else"""

# import jax as jnp
import numpy as np


def is_multiple(numbers, base):
    """
    For each member of `numbers`, checks if it is a multiple of `base`, returning True
    if it is the case.

    :param numbers: array of numbers
    :param base: single number
    :returns: boolean array with the same dimensions as `numbers`
    """
    divided_by_base = np.true_divide(numbers, base)
    floor_division = np.floor_divide(numbers, base)
    return np.isclose(divided_by_base, floor_division)
