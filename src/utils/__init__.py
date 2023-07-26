import logging
import sys
from enum import IntEnum
from typing import List

import random
import gurobipy as gp
from gurobipy import GRB

import torch.multiprocessing as mp

from .constants import VAR_NODE, CONS_NODE
from .config import NUM_WORKERS

# logging constants -- should we do more with the levels?
LOGGER_INITIALIZED: bool = False

WORKER_POOL_INITIALIZED: bool = False
WORKER_POOL = None

REPAIR_LEVEL: int = 5
REPAIR_NAME: str = 'REPAIR'

GP_ENV_INITIALIZED = False
GP_ENV = None

FORMAT_STR = "%(asctime)s %(levelname)s %(name)s.%(filename)s::%(funcName)s %(message)s"


def initialize_logger(filename: str = '',
                      level: int = logging.DEBUG):
    global LOGGER_INITIALIZED
    if not LOGGER_INITIALIZED:
        if str != '':
            logging.basicConfig(filename=filename, format=FORMAT_STR, level=level)
        else:
            logging.basicConfig(stream=sys.stdout, format=FORMAT_STR, level=level)
        LOGGER_INITIALIZED = True
        logging.addLevelName(REPAIR_LEVEL, REPAIR_NAME)


def get_global_pool():
    global WORKER_POOL_INITIALIZED, WORKER_POOL
    if WORKER_POOL_INITIALIZED:
        return WORKER_POOL
    else:
        WORKER_POOL_INITIALIZED = True
        WORKER_POOL = mp.Pool(NUM_WORKERS)
        return WORKER_POOL


def get_global_env():
    global GP_ENV_INITIALIZED, GP_ENV
    if GP_ENV_INITIALIZED:
        return GP_ENV
    else:
        GP_ENV_INITIALIZED = True
        GP_ENV = gp.Env()
        GP_ENV.setParam(GRB.Param.OutputFlag, 0)
        return GP_ENV


def range_permutation(n: int) -> List[int]:
    """
    Generates a permutation of the integers from 0 to n-1

    TODO: seed the rng so that we can replicate behavior
    :param n:
    :return:
    """
    numbers: List[int] = list(range(n))
    i: int
    for i in range(n):
        j: int = random.randint(i, n - 1)
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


class IndexEnum(IntEnum):

    def _generate_next_value_(name, start, count, last_values):
        start = 0
        return IntEnum._generate_next_value_(name, start, count, last_values)


def create_rng_seeds(num_trajectories):
    """Creates a list of rng_seeds
    :param num_trajectories:
    :return:
    """
    rng_seeds = [random.randint(0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff) for _ in range(num_trajectories)]
    return rng_seeds
