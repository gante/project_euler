""" Module with functions that won't fit anywhere else"""


def is_palindrome(number):
    """ Returns True if `number` is a palindrome, False otherwise.
    """
    num_str = str(number)
    num_comparisons = len(num_str) // 2
    for idx in range(num_comparisons):
        if num_str[idx] != num_str[-1-idx]:
            return False
    return True
