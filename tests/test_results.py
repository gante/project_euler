""" Tests whether the results of the problems are correct. That way, we can safely change
shared functions without breaking past solutions.

Recommended usage: run `pytest -v` while inside the repository. Because it may take up to a minute
per solved problem, use sparingly.

WARNING: don't read this file if you don't want to see the solutions to the problems. There
is no joy in that, am I right? ;)
"""

import os
import time
from datetime import timedelta
import importlib
import pytest

from jax.config import config
config.update("jax_enable_x64", True)   # Critical to test 64-bit dependent problems (e.g. prob. 3)


SOLUTIONS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..',
    'gante_project_euler',
    'solutions'
)
SOLUTIONS = {
    1: 233168,
    2: 4613732,
    3: 6857,
    4: 906609,
    5: 232792560,
}


def test_coverage():
    """ Tests whether all problems have a solution here. At most 1 problem may be missing its
    solution (the problem that is being worked on at the moment).
    """
    num_solutions = len(SOLUTIONS)
    num_problems = len(
        [f for f in os.listdir(SOLUTIONS_DIR) if os.path.isfile(os.path.join(SOLUTIONS_DIR, f))]
    )
    assert num_problems - num_solutions <= 1, \
        "Missing solutions! ({} problems, {} solutions)".format(num_problems, num_solutions)


@pytest.mark.parametrize("problem_idx", list(SOLUTIONS.keys()))
def test_results(problem_idx):
    """ Tests whether the solutions of the problems are correct and that they run within a minute.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    module_name = "gante_project_euler.solutions.problem_" + str(problem_idx)
    module = importlib.import_module(module_name)
    start = time.time()
    result = module.get_solution()
    end = time.time()
    assert result == SOLUTIONS[problem_idx], \
        "The result ({}) did not match the solution ({})!".format(result, SOLUTIONS[problem_idx])
    duration = timedelta(seconds=end-start)
    assert duration < timedelta(minutes=1), \
        "It took more than a minute to run! (duration = {})".format(duration)
