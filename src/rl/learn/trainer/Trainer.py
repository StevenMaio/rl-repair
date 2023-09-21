from abc import ABC, abstractmethod

from src.rl.utils import DataSet
from src.mip.heuristic import FixPropRepairLearn


class Trainer(ABC):

    @abstractmethod
    def train(self,
              fprl: FixPropRepairLearn,
              data_set: DataSet,
              model_output: str = None):
        ...
