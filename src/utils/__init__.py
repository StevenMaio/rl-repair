import logging
import sys

import random

# logging constants -- should we do more with the levels?
LOGGER_INITIALIZED: bool = False

REPAIR_LEVEL: int = 5
REPAIR_NAME: str = 'REPAIR'


def initialize_logger(filename: str = '',
                      level: int = logging.DEBUG):
    global LOGGER_INITIALIZED
    if not LOGGER_INITIALIZED:
        format_str = "%(levelname)s %(name)s.%(filename)s::%(funcName)s %(message)s"
        if str != '':
            logging.basicConfig(filename=filename, format=format_str, level=level)
        else:
            logging.basicConfig(stream=sys.stdout, format=format_str, level=level)
        LOGGER_INITIALIZED = True
        logging.addLevelName(REPAIR_LEVEL, REPAIR_NAME)


def range_permutation(n: int) -> list[int]:
    """
    Generates a permutation of the integers from 0 to n-1

    TODO: seed the rng so that we can replicate behavior
    :param n:
    :return:
    """
    numbers: list[int] = list(range(n))
    i: int
    for i in range(n):
        j: int = random.randint(i, n-1)
        numbers[i], numbers[j] = numbers[j], numbers[i]
    return numbers


def compute_interval_distance(lb, ub, x):
    """
    Computes the distance between a point x and the interval [lb, ub]
    :param lb:
    :param ub:
    :param x:
    :return:
    """
    if lb <= x <= ub:
        return 0
    else:
        return min(abs(x - lb), abs(x - ub))
