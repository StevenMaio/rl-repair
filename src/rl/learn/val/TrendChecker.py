from .ValidationProgressChecker import ValidationProgressChecker
from src.utils.timeseries import DesTimeSeries


class TrendChecker(ValidationProgressChecker):
    _initialized: bool

    _num_worse_iters: int
    _max_num_worse_iters: int

    _level_weight: float
    _trend_weight: float
    _init_trend: float
    _time_series: DesTimeSeries

    def __init__(self,
                 max_num_worse_iters,
                 level_weight,
                 trend_weight,
                 init_trend=0):
        self._num_worse_iters = 0
        self._max_num_worse_iters = max_num_worse_iters
        self._level_weight = level_weight
        self._trend_weight = trend_weight
        self._init_trend = init_trend
        self._initialized = False

    def update_progress(self, val_score: float):
        if self._initialized:
            self._time_series.add(val_score)
            if self._time_series.trend < 0:
                self._num_worse_iters += 1
            else:
                self._num_worse_iters = 0
        else:
            self._initialized = True
            self._time_series = DesTimeSeries(self._level_weight,
                                              self._trend_weight,
                                              val_score,
                                              self._init_trend)

    def continue_training(self) -> bool:
        return self._num_worse_iters < self._max_num_worse_iters

    def reset(self):
        self._initialized = False
        self._num_worse_iters = 0

    def corrected_score(self) -> float:
        return self._time_series.level
