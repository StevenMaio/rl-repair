from src.utils.config import PARAMS

from .FirstOrderMethod import FirstOrderMethod
from .Adam import Adam
from .GradientAscent import GradientAscent

optimizers = {
    'Adam': Adam,
    'GradientAscent': GradientAscent
}


def optimizer_fom_config(config: dict):
    name = config["class"]
    return optimizers[name](**config[PARAMS])
