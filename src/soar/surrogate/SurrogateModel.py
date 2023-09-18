"""
The abstract base class for surrogate models. See [1] for more details.

References:
    [1] L. Mathesen, G. Pedrielli, S. H. Ng, and Z. B. Zabinsky, “Stochastic optimization
    with adaptive restart: a framework for integrated local and global learning,”
    J Glob Optim, vol. 79, no. 1, pp. 87–110, Jan. 2021, doi: 10.1007/s10898-020-00937-5.
"""
from abc import ABC, abstractmethod
from typing import Union

import torch


class SurrogateModel(ABC):

    @abstractmethod
    def init(self, data_points: torch.Tensor, observations: torch.Tensor):
        ...

    @abstractmethod
    def add_point(self, x: torch.Tensor, y: Union[torch.Tensor, float]):
        ...

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def predict_var(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def reset(self):
        ...

    @property
    @abstractmethod
    def support(self):
        raise Exception("support not implemented")

    @property
    @abstractmethod
    def mean_estimate(self):
        raise Exception("mean_estimate not implemented")

    @property
    @abstractmethod
    def var_estimate(self):
        raise Exception("var_estimate not implemented")

    @property
    @abstractmethod
    def observations(self) -> torch.Tensor:
        raise Exception("observations not implemented")

    @property
    @abstractmethod
    def data_points(self) -> torch.Tensor:
        raise Exception("data_points not implemented")
