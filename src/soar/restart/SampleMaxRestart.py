"""
A preliminary restart mechanism. The details are available in [2]. This restart
mechanism requires second order. Thus, it should only be called with SurrogateModel
instances that implement the predictive mean estimate and predictive var estimate.

References:
    [1] L. Mathesen, G. Pedrielli, S. H. Ng, and Z. B. Zabinsky, “Stochastic optimization
        with adaptive restart: a framework for integrated local and global learning,"
        J Glob Optim, vol. 79, no. 1, pp. 87–110, Jan. 2021, doi: 10.1007/s10898-020-00937-5.
    [2] SOAR Implementation Note. [[SOAR Implementation#Prelminary Restart Procedure]]. My
        obsidian vault.
"""
import torch

from .RestartMechanism import RestartMechanism
from ..surrogate import SurrogateModel


class SampleMaxRestart(RestartMechanism):
    _best_ei_threshold: float
    _num_restart_samples: int
    _noise_parameter: float
    _std_normal_distr: torch.distributions.Normal

    def __init__(self,
                 best_ei_threshold: float,
                 num_restart_samples: int,
                 noise_parameter: float = 0.0):
        self._best_ei_threshold = best_ei_threshold
        self._num_restart_samples = num_restart_samples
        self._noise_parameter = noise_parameter
        self._std_normal_distr = torch.distributions.Normal(0.0, 1.0)

    def determine_restart_point(self, surrogate_model: SurrogateModel) -> torch.Tensor:
        """
        Determines a new restart location according to the details in [2].

        :return: a restart location for local search
        """
        support = surrogate_model.support
        num_dimensions = support.shape[0]
        scaling_factor = support[:, 1] - support[:, 0]
        samples = torch.rand((self._num_restart_samples, num_dimensions))
        restart_points = samples * scaling_factor + support[:, 0]

        predictions = torch.zeros(self._num_restart_samples)
        predictive_var = torch.zeros(self._num_restart_samples)
        for i, x in enumerate(restart_points):
            predictions[i] = surrogate_model.predict(x)
            predictive_var[i] = surrogate_model.predict_var(x)

        best_val = surrogate_model.observations.max()
        mean_diff = predictions - best_val
        normalized_diff = mean_diff / predictive_var

        expected_change = mean_diff * self._std_normal_distr.cdf(normalized_diff) \
                          + predictive_var * torch.exp(self._std_normal_distr.log_prob(normalized_diff))
        expected_improvement = torch.fmax(expected_change, torch.zeros(1))
        best_ei = expected_improvement.max()
        options = torch.where(expected_improvement >= self._best_ei_threshold * best_ei, best_ei, 0)
        indices = torch.nonzero(options).flatten()
        if len(indices) > 0:
            samples = samples[indices, :]

        num_data_points = len(surrogate_model.data_points)
        num_samples = len(samples)

        temp = (samples.view(1, num_samples, num_dimensions)
                - surrogate_model.data_points.view(num_data_points, 1, num_dimensions))
        temp = temp.square().sum(axis=2)
        largest_distances = temp.amax(axis=0)
        best_idx = largest_distances.argmax()
        start_point = samples[best_idx].clone()
        return start_point

    def get_cost(self) -> int:
        return 1
