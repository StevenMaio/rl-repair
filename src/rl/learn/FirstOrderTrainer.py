import torch

from .FirstOrderMethod import FirstOrderMethod
from .GradientEstimator import GradientEstimator

from src.mip.heuristic import FixPropRepairLearn

import logging


class FirstOrderTrainer:
    _optimization_method: FirstOrderMethod
    _gradient_estimator: GradientEstimator
    _num_epochs: int
    _logger: logging.Logger

    def __init__(self,
                 optimization_method: FirstOrderMethod,
                 gradient_estimator: GradientEstimator,
                 num_epochs: int):
        self._optimization_method = optimization_method
        self._gradient_estimator = gradient_estimator
        self._num_epochs = num_epochs
        self._logger = logging.getLogger(__name__)

    def train(self,
              fprl: FixPropRepairLearn,
              training_instances,
              save_rate: int = float('inf'),
              model_output: str = None):
        self._optimization_method.reset()
        policy_architecture = fprl.policy_architecture
        for epoch in range(self._num_epochs):
            gradient_estimate = self._gradient_estimator.estimate_gradient(training_instances,
                                                                           fprl)
            self._optimization_method.step(fprl.policy_architecture,
                                           gradient_estimate)
            # save model at epoch intervals
            if model_output is not None and (epoch + 1) % save_rate == 0:
                torch.save(policy_architecture.state_dict(), model_output)
        # save model at end
        if model_output is not None:
            torch.save(policy_architecture.state_dict(), model_output)
