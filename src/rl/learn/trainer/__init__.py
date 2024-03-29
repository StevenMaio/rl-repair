from .Trainer import Trainer
from .FirstOrderTrainer import FirstOrderTrainer

from src.soar import SOAR

trainers = {
    'SOAR': SOAR,
    'FirstOrderTrainer': FirstOrderTrainer
}


def trainer_from_config(config: dict):
    name = config["class"]
    trainer = trainers[name].from_config(config)
    return trainer
