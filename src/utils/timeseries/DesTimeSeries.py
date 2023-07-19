"""
A double exponential time series representation
"""


class DesTimeSeries:
    _level: float
    _trend: float
    _level_weight: float
    _trend_weight: float

    def __init__(self, level_decay, trend_decay, init_level=0, init_trend=0):
        self._level = init_level
        self._trend = init_trend
        self._level_weight = level_decay
        self._trend_weight = trend_decay

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
