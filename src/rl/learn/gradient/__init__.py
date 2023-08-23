from src.utils.config import PARAMS

from .GradientEstimator import GradientEstimator
from .EsParallelTrajectories import EsParallelTrajectories
from .EsParallelInstances import EsParallelInstances
from .EvolutionaryStrategiesSerial import EvolutionaryStrategiesSerial
from .PolicyGradientSerial import PolicyGradientSerial
from .PolicyGradientParallel import PolicyGradientParallel

estimators = {
    'EsParallelTrajectories': EsParallelTrajectories,
    'EsParallelInstances': EsParallelInstances,
    'EvolutionaryStrategiesSerial': EvolutionaryStrategiesSerial,
    'PolicyGradientSerial': PolicyGradientSerial,
    'PolicyGradientParallel': PolicyGradientParallel
}


def gradient_estimator_from_config(config: dict):
    name = config["class"]
    return estimators[name](**config[PARAMS])
