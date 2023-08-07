"""
A double exponential time series representation
"""
from .TimeSeries import TimeSeries


class DesTimeSeries(TimeSeries):
    _level: float
    _trend: float
    _level_weight: float
    _trend_weight: float
    _init_level: float
    _init_trend: float

    def __init__(self, level_decay, trend_decay, init_level=0, init_trend=0):
        self._level = init_level
        self._trend = init_trend
        self._level_weight = level_decay
        self._trend_weight = trend_decay
        self._init_level = init_level
        self._init_trend = init_trend

    def add(self, data) -> float:
        new_level = self._level_weight * data + (1 - self._level_weight) * (self._level + self._trend)
        new_trend = self._trend_weight * (new_level - self._level) + (1 - self._trend_weight) * self._trend
        self._level = new_level
        self._trend = new_trend
        return new_level

    @property
    def level(self):
        return self._level

    @property
    def trend(self):
        return self._trend

    def reset(self, hard_reset=False):
        self._level = self._init_level
        self._trend = self._init_trend
