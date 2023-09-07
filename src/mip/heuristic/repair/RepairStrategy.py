from abc import ABC, abstractmethod

from typing import List, Tuple, Any


class RepairStrategy(ABC):
    name: str

    @abstractmethod
    def repair_domain(self,
                      model: "Model",
                      repair_changes: List["DomainChange"],
                      generator=None) -> bool:
        """
        Attempts to repair the domain of architecture. If the repair is successful, then
        a boolean value of True is returned, and the repair changes are added to
        the list `repair_changes`
        :param generator:
        :param model:
        :param repair_changes:
        :return:
        """
        ...

    @property
    def num_moves(self):
        raise NotImplementedError("not implemented")

    @abstractmethod
    def find_shift_candidates(self,
                              model: "Model",
                              constraint: "Constraint") -> Tuple[List[Tuple[Any, Any, float]], bool]:
        ...
