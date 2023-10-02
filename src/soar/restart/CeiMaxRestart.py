"""
The crowded expectation improvement restart mechanism introduced in [1]. This
restart mechanism involves solving two optimization problems. See [1] for the details.

This is based on the surrogate model

References:
    [1] L. Mathesen, G. Pedrielli, S. H. Ng, and Z. B. Zabinsky, “Stochastic optimization
        with adaptive restart: a framework for integrated local and global learning,"
        J Glob Optim, vol. 79, no. 1, pp. 87–110, Jan. 2021, doi: 10.1007/s10898-020-00937-5.
"""
import torch
import numpy as np

from scipy.stats import norm
from scipy.optimize import minimize

from .RestartMechanism import RestartMechanism
from ..surrogate import SurrogateModel


class CeiMaxRestart(RestartMechanism):
    _tolerance: float

    def __init__(self, optimality_tolerance: float):
        self._tolerance = optimality_tolerance

    def determine_restart_point(self, surrogate_model: SurrogateModel) -> torch.Tensor:
        best_obs = surrogate_model.observations.min()

        def expected_improvement(x: np.ndarray):
            x_torch = torch.from_numpy(x)
            temp = torch.zeros(surrogate_model.num_dimensions)
            temp[:6] = x_torch
            pred_val = surrogate_model.predict(temp)
            pred_std_dev = surrogate_model.predict_var(temp).sqrt()
            normalized_diff = (pred_val - best_obs) / pred_std_dev
            temp = (pred_val - best_obs) * norm.cdf(normalized_diff)
            temp += pred_std_dev * norm.pdf(normalized_diff)
            return -np.amax((0.0, temp))

        np_init_point = np.zeros(6)
        res = minimize(expected_improvement,
                       np_init_point,
                       method='Nelder-Mead',
                       options={
                           'maxiter': 200,
                           'adaptive': True,
                       },
                       bounds=[(-100, 100)])
        x = 3

    def get_cost(self) -> int:
        pass
