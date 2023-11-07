from src.utils.config import PARAMS

from .FixingOrderStrategy import FixingOrderStrategy
from .LeftRightOrder import LeftRightOrder
from .RandomFixingOrder import RandomFixingOrder
from .TypeFixingOrder import TypeFixingOrder
from .VariableLocksOrder import VariableLocksOrder

fixing_order_strategies = {
    'RandomFixingOrder': RandomFixingOrder,
    'LeftRightOrder': LeftRightOrder,
    'TypeFixingOrder': TypeFixingOrder,
    'VariableLocksOrder': VariableLocksOrder,
}


def fixing_order_strategy_from_config(config: dict):
    name = config["class"]
    return fixing_order_strategies[name](**config[PARAMS])
