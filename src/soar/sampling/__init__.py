from src.utils.config import PARAMS
from .ExperimentalDesign import ExperimentalDesign
from .LatinHypercube import LatinHypercube

experimental_designs = {
    'LatinHypercube': LatinHypercube
}

def experimental_design_from_config(config: dict):
    name = config["class"]
    return experimental_designs[name](**config[PARAMS])