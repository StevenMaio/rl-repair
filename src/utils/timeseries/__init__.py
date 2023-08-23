from .TimeSeries import TimeSeries

from .DesTimeSeries import DesTimeSeries
from .KMovingMeans import KMovingMeans
from ..config import PARAMS

time_series = {
    'DesTimeSeries': DesTimeSeries,
    'KMovingMeans': KMovingMeans
}


def time_series_from_config(config: dict):
    name = config["class"]
    return time_series[name](**config[PARAMS])
