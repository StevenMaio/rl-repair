from abc import ABC, abstractmethod
from src.soar.objective import Objective


class LocalSearch(ABC):

    @abstractmethod
    def init(self, start_point, objective: Objective):
        ...

    @abstractmethod
    def step(self, objective):
        ...

    @abstractmethod
    def objective_value(self):
        ...

    @abstractmethod
    def current_point(self):
        ...

    @abstractmethod
    def computation_cost(self):
        ...
