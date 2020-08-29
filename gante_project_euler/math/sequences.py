""" Module for numeric sequences (e.g. Fibonacci)"""


def fibonacci_up_to(*, index=None, value=None):
    """
    [NP] Returns all fibonacci numbers up to the `index`-th member, or until the value exceeds
    `value`.

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
        all_fibonacci += [new_member]
    return all_fibonacci


def triangle_up_to(*, index=None, value=None):
    """
    [NP] Returns all triangle numbers up to the `index`-th member, or until the value exceeds
    `value`.

    :param index: integer
    :param value: integer
    :returns: list of integers, with the desired triangle sequence numbers
    """
    assert index is not None or value is not None, "One named input must be given"
    all_triangle = []
    loop_length = index or 2**31
    new_member = 0
    for i in range(1, loop_length):
        new_member += i
        if value and new_member > value:
            break
        all_triangle.append(new_member)
    return all_triangle
