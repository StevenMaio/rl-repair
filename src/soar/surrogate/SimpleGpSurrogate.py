"""
An implementation of the Guassian Process surrogate model introduced in [1].
Details are provided in the appendix. This Gaussian process is stationary and
assumes to know the true correlation function of the Gaussian Process.

We note that the correlation function used in based on the exponential correlation
function defined on p. 64 of [2]. This choice is made for consistency, as potential
future work may involve using estimators that are based on this definition.

References:
    [1] L. Mathesen, G. Pedrielli, S. H. Ng, and Z. B. Zabinsky, “Stochastic optimization
    with adaptive restart: a framework for integrated local and global learning,”
    J Glob Optim, vol. 79, no. 1, pp. 87–110, Jan. 2021, doi: 10.1007/s10898-020-00937-5.
    [2] T. J. Santner, B. J. Williams, and W. I. Notz, The Design and Analysis of
    Computer Experiments. in Springer Series in Statistics. New York, NY: Springer
    New York, 2003. doi: 10.1007/978-1-4757-3799-8.

"""
import torch

from src.rl.utils import TensorList

from typing import List

DEFAULT_SIZE_INCREMENT = 20


class SimpleGpSurrogate:
    _mean_estimate: torch.Tensor
    _var_estimate: torch.Tensor
    _data_points: List[TensorList]
    _observations: torch.Tensor

    _corr_matrix: torch.Tensor
    _corr_inv: torch.Tensor
    _corr_parameters: TensorList

    # related to dynamic sizing
    _max_size: int
    _size: int

    def __init__(self,
                 correlation_parameters: TensorList,
                 max_size: int):
        self._mean_estimate = torch.zeros(1)
        self._var_estimate = torch.ones(1)
        self._data_points = []
        self._observations = torch.zeros(max_size)
        self._corr_matrix = torch.zeros((max_size, max_size))
        self._corr_inv = None
        self._corr_parameters = correlation_parameters

        self._max_size = max_size
        self._size = 0

    def predict(self, x):
        """
        Evaluates the estimator function at x
        :param x:
        :return:
        """
        r = self._compute_corr_vector(x)
        estimate = (self._mean_estimate +
                    r.T @ self._corr_inv @ (self._observations[:self._size] - self._mean_estimate))
        return estimate

    def predict_var(self, x):
        """
        Evaluates the estimator function at x
        :param x:
        :return:
        """
        r = self._compute_corr_vector(x)
        ones = torch.ones(self._size)
        temp = (1 - ones.T @ self._corr_inv @ r).square()
        temp /= ones.T @ self._corr_inv @ ones
        temp += 1 - r.T @ self._corr_inv @ r
        var_estimate = self._var_estimate * temp
        return var_estimate

    def add_point(self, x, y):
        if self._size == self._max_size:
            self._resize_data_tensors(self._max_size + DEFAULT_SIZE_INCREMENT)
        self._observations[self._size] = y

        # update the current estimators and data
        r = self._compute_corr_vector(x)
        self._corr_matrix[self._size, self._size] = 1.0
        self._corr_matrix[self._size, :self._size] = r
        self._corr_matrix[:self._size, self._size] = r

        self._size += 1
        self._data_points.append(x)

        self._corr_inv = torch.linalg.inv(self._corr_matrix[:self._size, :self._size])

        # update the moment estimates
        corr_obs_sum = torch.ones(self._size).T @ self._corr_inv @ self._observations[:self._size]
        self._mean_estimate = corr_obs_sum / (torch.ones(self._size).T @ self._corr_inv @ torch.ones(self._size))
        mean_diff = self._observations[:self._size] - self._mean_estimate * torch.ones(self._size)
        self._var_estimate = mean_diff.T @ self._corr_inv @ mean_diff / self._size

    def init(self, data_points, observations):
        if len(data_points) > self._max_size:
            self._resize_data_tensors(len(data_points) + DEFAULT_SIZE_INCREMENT)
        for i, (x, y) in enumerate(zip(data_points, observations)):
            for j, v in enumerate(data_points):
                if i == j:
                    self._corr_matrix[i, j] = 1.0
                else:
                    corr = self._compute_correlation(x, v)
                    self._corr_matrix[i, j] = corr
                    self._corr_matrix[j, i] = corr
            self._observations[i] = y
            self._data_points.append(x)
        self._size = len(data_points)
        self._corr_inv = torch.linalg.inv(self._corr_matrix[:self._size, :self._size])

        # compute the moment estimates
        corr_obs_sum = torch.ones(self._size).T @ self._corr_inv @ self._observations[:self._size]
        self._mean_estimate = corr_obs_sum / (torch.ones(self._size).T @ self._corr_inv @ torch.ones(self._size))

        mean_diff = self._observations[:self._size] - self._mean_estimate * torch.ones(self._size)
        self._var_estimate = mean_diff.T @ self._corr_inv @ mean_diff / self._size

    def _compute_corr_vector(self, x):
        """
        Computes the vector of correlations with respect to the current data points.
        :param x:
        :return:
        """
        r = torch.zeros(self._size)
        for i, u in enumerate(self._data_points):
            r[i] = self._compute_correlation(x, u)
        return r

    def _compute_correlation(self, u: TensorList, v: TensorList) -> torch.Tensor:
        """
        Computes the correlation between parameters u and v.

        :param u:
        :param v:
        :return:
        """
        correlation = torch.ones(1)
        for t1, t2, theta in zip(u, v, self._corr_parameters):
            correlation *= torch.exp(-((t1 - t2) / theta).square()).prod()
        return correlation

    def _resize_data_tensors(self, new_size: int):
        """
        Reallocates the correlation matrix and observation vector. Then updates
        the maximum size field.
        :param new_size:
        :return:
        """
        if new_size < self._size:
            raise Exception("new size is less than the number of observations")
        new_obs = torch.zeros(new_size)
        new_corr_matrix = torch.zeros((new_size, new_size))
        new_obs[:self._size] = self._observations
        new_corr_matrix[:self._size, :self._size] = self._corr_matrix
        self._observations = new_obs
        self._corr_matrix = new_corr_matrix
        self._max_size = new_size

    @property
    def mean_esimate(self):
        return self._mean_estimate

    @property
    def var_estimate(self):
        return self._var_estimate
