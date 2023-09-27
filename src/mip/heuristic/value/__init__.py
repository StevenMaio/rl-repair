from src.utils.config import PARAMS

from .ValueFixingStrategy import ValueFixingStrategy
from .RandomValueFixing import RandomValueFixing
from .UpperBoundFirst import UpperBoundFirst
from .GoodOjbective import GoodObjective
from .BadObjective import BadObjective
from .LpFixingStrategy import LpFixingStrategy
from .SmallestViolationStrategy import SmallestViolationStrategy

value_fixing_strategies = {
    'RandomValueFixing': RandomValueFixing,
    'UpperBoundFirst': UpperBoundFirst,
    'GoodOjbective': GoodObjective,
    'BadObjective': BadObjective,
    'LpFixingStrategy': LpFixingStrategy,
    'SmallestViolationStrategy': SmallestViolationStrategy
}


def value_fixing_strategy_from_config(config: dict):
    name = config["class"]
    return value_fixing_strategies[name](**config[PARAMS])
