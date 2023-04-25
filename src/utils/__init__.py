import logging
import sys

import random

from typing import Iterator


LOGGER_INITIALIZED: bool = False


def initialize_logger(filename: str = '', level: int = logging.DEBUG):
    global LOGGER_INITIALIZED
    if not LOGGER_INITIALIZED:
        format_str = "%(levelname)s %(name)s.%(filename)s::%(funcName)s %(message)s"
        if str != '':
            logging.basicConfig(filename=filename, format=format_str, level=level)
        else:
            logging.basicConfig(stream=sys.stdout, format=format_str, level=level)
        LOGGER_INITIALIZED = True


def permutation_iter(n: int) -> Iterator[int]:
    numbers: list[int] = list(range(n))
    i: int
    for i in range(n):
        j: int = random.randint(i, n-1)
        numbers[i], numbers[j] = numbers[j], numbers[i]
        yield numbers[i]
