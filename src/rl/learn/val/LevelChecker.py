from .ValidationProgressChecker import ValidationProgressChecker
from src.utils.timeseries import DesTimeSeries


class LevelChecker(ValidationProgressChecker):
    _initialized: bool

    _best_level: float
    _num_worse_iters: int
    _max_num_worse_iters: int

    _level_weight: float
    _trend_weight: float
    _init_trend: float
    _time_series: DesTimeSeries

    def __init__(self,
                 max_num_worse_iters,
                 time_series
                 ):
        self._num_worse_iters = 0
        self._max_num_worse_iters = max_num_worse_iters
        self._best_level = 0
        self._time_series = time_series

    def update_progress(self, val_score: float):
        self._time_series.add(val_score)
        if self._time_series.level < self._best_level:
            self._num_worse_iters += 1
        else:
            self._best_level = self._time_series.level
            self._num_worse_iters = 0

    def continue_training(self) -> bool:
        return self._num_worse_iters < self._max_num_worse_iters

    def reset(self):
        self._initialized = False
        self._num_worse_iters = 0
        self._time_series.reset()

    def soft_reset(self):
        ...

    def corrected_score(self) -> float:
        return self._time_series.level
