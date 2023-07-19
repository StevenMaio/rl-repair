from abc import ABC, abstractmethod


class ValidationProgressChecker(ABC):

    @abstractmethod
    def update_progress(self, val_score: float):
        ...

    @abstractmethod
    def continue_training(self) -> bool:
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def corrected_score(self):
        ...
