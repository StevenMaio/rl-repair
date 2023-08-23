from src.utils.config import PARAMS, TIME_SERIES_CONFIG

from .ValidationProgressChecker import ValidationProgressChecker

from .GreedyChecker import GreedyChecker
from .TrendChecker import TrendChecker
from .LevelChecker import LevelChecker

from src.utils.timeseries import time_series_from_config

progress_checkers = {
    'TrendChecker': TrendChecker,
    'LevelChecker': LevelChecker,
}


def progress_checker_from_config(config: dict):
    name = config["class"]
    params = config[PARAMS]
    if "time_series" in params:
        time_series = time_series_from_config(params[TIME_SERIES_CONFIG])
        params[TIME_SERIES_CONFIG] = time_series
    return progress_checkers[name](**params)
