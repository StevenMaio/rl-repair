from src.utils.config import PARAMS
from .RestartMechanism import RestartMechanism
from .SampleMaxRestart import SampleMaxRestart
from .CeiMaxRestart import CeiMaxRestart

restart_mechanisms = {
    'SampleMaxRestart': SampleMaxRestart,
    'CeiMaxRestart': CeiMaxRestart
}

def restart_mechanism_from_config(config: dict):
    name = config["class"]
    return restart_mechanisms[name](**config[PARAMS])
