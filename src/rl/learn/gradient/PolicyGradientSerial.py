import logging
import torch

import gurobipy as gp
from gurobipy import GRB

from .GradientEstimator import GradientEstimator
from rl.mip import EnhancedModel
from rl.utils import TensorList

from .PolicyGradientHelper import PolicyGradientHelper


class PolicyGradientSerial(GradientEstimator):
    _num_successes: int

    _num_trajectories: int
    _batch_size: int
    _use_all_samples: bool

    _logger: logging.Logger

    def __init__(self,
                 num_trajectories: int,
                 batch_size: int = float('inf')):
        self._num_trajectories = num_trajectories
        self._num_trajectories = num_trajectories
        if batch_size == float('inf'):
            self._use_all_samples = True
        else:
            self._use_all_samples = False
        self._batch_size = batch_size
        self._logger = logging.getLogger(__package__)

    def estimate_gradient(self, instances, fprl):
        policy_architecture = fprl.policy_architecture
        gradient_estimate = TensorList.zeros_like(policy_architecture.parameters())
        self._num_successes = 0
        if self._use_all_samples:
            batch = instances
            batch_size = len(instances) * self._num_trajectories
        else:
            indices = torch.randint(len(instances), (self._batch_size, ))
            batch = [instances[i] for i in indices]
            batch_size = self._batch_size * self._num_trajectories
        self._compute_batch_gradient(fprl, batch, gradient_estimate)
        self._logger.info('END_GRADIENT_COMPUTATION success_rate=%.2f', self._num_successes / batch_size)
        return gradient_estimate

    def _compute_batch_gradient(self, fprl, batch, gradient_estimate):
        env = gp.Env()
        env.setParam(GRB.Param.OutputFlag, 0)
        for instance in batch:
            self._compute_instance_gradient_estimate(fprl,
                                                     env,
                                                     instance,
                                                     gradient_estimate)

    def _compute_instance_gradient_estimate(self,
                                            fprl,
                                            env,
                                            instance,
                                            gradient_estimate):
        policy_architecture = fprl.policy_architecture
        for trajectory_num in range(self._num_trajectories):
            gp_model = gp.read(instance, env)
            model = EnhancedModel.from_gurobi_model(gp_model,
                                                    gnn=policy_architecture.gnn,
                                                    convert_ge_cons=True)
            fprl.find_solution(model)
            reward = fprl.reward
            model.reset()
            if reward > 0:
                helper = PolicyGradientHelper(fprl, model)
                gradient_estimate.add_to_self(helper.compute_gradient_estimate())
                self._num_successes += 1
