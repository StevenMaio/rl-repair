from src.utils.config import PARAMS
from .TerminationMechanism import TerminationMechanism
from .AtMostKIters import AtMostKIters

termination_mechanisms = {
    'AtMostKIters': AtMostKIters
}


def termination_mechanism_from_config(config: dict):
    name = config["class"]
    return termination_mechanisms[name](**config[PARAMS])
