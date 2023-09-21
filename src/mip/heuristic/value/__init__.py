from src.utils.config import PARAMS

from .ValueFixingStrategy import ValueFixingStrategy
from .RandomValueFixing import RandomValueFixing
from .UpperBoundFirst import UpperBoundFirst

value_fixing_strategies = {
    'RandomValueFixing': RandomValueFixing,
    'UpperBoundFirst': UpperBoundFirst
}


def value_fixing_strategy_from_config(config: dict):
    name = config["class"]
    return value_fixing_strategies[name](**config[PARAMS])
