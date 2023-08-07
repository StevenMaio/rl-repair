from .TimeSeries import TimeSeries
from typing import List


class KMovingMeans(TimeSeries):

    def reset(self, hard_reset=False):
        self._level = 0
        self._data[self._current_idx] = 0

    _k: int
    _data: List[float]
    _level: float
    _current_idx: int

    def __init__(self, k):
        self._data = [0] * k
        self._current_idx = 0
        self._level = 0
        self._k = k

    def add(self, data):
        self._level += (data - self._data[self._current_idx]) / self._k
        self._data[self._current_idx] = data
        self._current_idx = (self._current_idx + 1) % self._k
        return self._level

    @property
    def level(self):
        return self._level
