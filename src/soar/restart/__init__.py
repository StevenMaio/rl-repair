from src.utils.config import PARAMS
from .RestartMechanism import RestartMechanism
from .SampleMaxRestart import SampleMaxRestart

restart_mechanisms = {
    'SampleMaxRestart': SampleMaxRestart
}

def restart_mechanism_from_config(config: dict):
    name = config["class"]
    return restart_mechanisms[name](**config[PARAMS])
