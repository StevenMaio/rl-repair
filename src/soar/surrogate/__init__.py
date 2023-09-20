from src.utils.config import PARAMS
from .SurrogateModel import SurrogateModel
from .SimpleGpSurrogate import SimpleGpSurrogate

surrogates = {
    'SimpleGpSurrogate': SimpleGpSurrogate
}


def surrogate_from_config(config: dict):
    name = config["class"]
    return surrogates[name](**config[PARAMS])
