""" Module for numeric sequences (e.g. Fibonacci)"""


def fibonacci_up_to(*, index=None, value=None):
    """
    Returns all fibonacci numbers up to the `index`-th member, or until the value exceeds `value`

    NOTE: this is a sequential function, not implemented in jax.

    :param index: integer
    :param value: integer
    :returns: list of integers, with the desired fibonacci numbers
    """
    assert index is not None or value is not None, "One named input must be given"
    all_fibonacci = []
    loop_length = index or 2**31
    for i in range(loop_length):
        if i == 0:
            new_member = 1
        elif i == 1:
            new_member = 2
        else:
            new_member = all_fibonacci[-1] + all_fibonacci[-2]
        if value and new_member > value:
            break
        else:
            all_fibonacci += [new_member]
    return all_fibonacci
