import logging
import sys
from enum import IntEnum
from .LogEvent import LogEvent

import torch
import torch.multiprocessing as mp

from .constants import VAR_NODE, CONS_NODE
from .config import NUM_WORKERS

# logging constants -- should we do more with the levels?
LOGGER_INITIALIZED: bool = False

WORKER_POOL_INITIALIZED: bool = False
WORKER_POOL = None

REPAIR_LEVEL: int = 5
REPAIR_NAME: str = 'REPAIR'

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


def initialize_global_pool(num_workers):
    global WORKER_POOL_INITIALIZED, WORKER_POOL
    if not WORKER_POOL_INITIALIZED:
        mp.set_start_method('forkserver')
        WORKER_POOL = mp.Pool(num_workers)
        WORKER_POOL_INITIALIZED = True
        return WORKER_POOL


def get_global_pool():
    global WORKER_POOL_INITIALIZED, WORKER_POOL
    if WORKER_POOL_INITIALIZED:
        return WORKER_POOL
    else:
        raise Exception('worker pool not initialized')


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


def create_rng_seeds(num_seeds):
    """Creates a list of rng_seeds
    :param num_seeds:
    :return:
    """
    rng_seeds = [torch.randint(0x8000_0000, 0xffff_ffff, (1,)) for _ in range(num_seeds)]
    return rng_seeds
