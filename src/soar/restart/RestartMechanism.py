"""
The abstract base class for the restart mechanism. For a detailed overview, see [1].

References:
    [1] L. Mathesen, G. Pedrielli, S. H. Ng, and Z. B. Zabinsky, “Stochastic optimization
        with adaptive restart: a framework for integrated local and global learning,"
        J Glob Optim, vol. 79, no. 1, pp. 87–110, Jan. 2021, doi: 10.1007/s10898-020-00937-5.
"""
import torch
from abc import ABC, abstractmethod

from ..surrogate import SurrogateModel


class RestartMechanism(ABC):

    @abstractmethod
    def determine_restart_point(self, surrogate_model: SurrogateModel) -> torch.Tensor:
        ...
