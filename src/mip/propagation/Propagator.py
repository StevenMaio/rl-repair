from abc import ABC, abstractmethod

from ..model import Model, Constraint, DomainChange


class Propagator(ABC):

    @abstractmethod
    def propagate(self,
                  model: Model,
                  constraint: Constraint,
                  domain_changes: list[DomainChange]):
        """
        Deduces domain changes based on the given constraint. Domain changes are
        appended to the domain_changes list, but not applied.
        :param model:
        :param constraint:
        :param domain_changes:
        :return:
        """
        ...
