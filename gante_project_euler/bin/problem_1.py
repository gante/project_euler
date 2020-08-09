""" Solution for Project Euler's problem #1 """

import time
from datetime import timedelta
import numpy as np
from gante_project_euler.math.misc import is_multiple


def main():
    """ Solves the problem and prints the answer. May print progress-related messages.
    """
    integers = np.arange(1000)
    multiples_of_3_or_5 = np.logical_or(
        is_multiple(integers, 3),
        is_multiple(integers, 5)
    )
    sum_of_multiples = np.sum(integers[multiples_of_3_or_5])
    return sum_of_multiples


if __name__ == "__main__":
    start = time.time()
    solution = main()
    end = time.time()
    print("Solution: {}".format(solution))
    print("Elapsed time: {} (HH:MM:SS.us)".format(timedelta(seconds=end-start)))
