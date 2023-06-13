from abc import ABC, abstractmethod

from typing import List


class RepairStrategy(ABC):
    name: str

    @abstractmethod
    def repair_domain(self, model: "Model", repair_changes: List["DomainChange"]) -> bool:
        """
        Attempts to repair the domain of architecture. If the repair is successful, then
        a boolean value of True is returned, and the repair changes are added to
        the list `repair_changes`
        :param model:
        :param repair_changes:
        :return:
        """
        ...
