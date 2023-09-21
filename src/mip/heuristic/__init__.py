from src.utils.config import CLASS

from .FixPropRepair import FixPropRepair
from .FixPropRepairLearn import FixPropRepairLearn

heuristics = {
    'FixPropRepair': FixPropRepair,
    'FixPropRepairLearn': FixPropRepairLearn
}


def heuristic_from_config(config: dict):
    name = config[CLASS]
    return heuristics[name].from_config(config)
