"""
An implementation of policy gradient in parallel. Unfortunately, this does not
entirely avoid serial computation of the gradient approximation. This model
runs the instances in parallel, and then returns the reward, the action history
and the instance. If an instance is successfully solved, then a history is returned.

At the end, all histories are processed in serial.
"""
import itertools
import logging
import torch

import torch.multiprocessing as mp

import gurobipy as gp
from gurobipy import GRB

from src.utils import get_global_pool
from src.utils.config import NUM_THREADS
from .GradientEstimator import GradientEstimator
from rl.mip import EnhancedModel
from rl.utils import TensorList

from .PolicyGradientHelper import PolicyGradientHelper


def _run_parallel_trajectories(fprl, instance, gradient_estimate):
    torch.set_num_threads(NUM_THREADS)
    env = gp.Env()
    env.setParam(GRB.Param.OutputFlag, 0)
    policy_architecture = fprl.policy_architecture
    gp_model = gp.read(instance, env)
    model = EnhancedModel.from_gurobi_model(gp_model,
                                            gnn=policy_architecture.gnn,
                                            convert_ge_cons=True)
    fprl.find_solution(model)
    reward = fprl.reward
    model.reset()
    if reward > 0:
        return reward, fprl.action_history, instance
    return reward, None, None


class PolicyGradientParallel(GradientEstimator):
    _num_successes: int

    _num_trajectories: int
    _batch_size: int
    _use_all_samples: bool

    _logger: logging.Logger

    _worker_pool: mp.Pool

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
        self._worker_pool = get_global_pool()

    def estimate_gradient(self, instances, fprl):
        policy_architecture = fprl.policy_architecture
        gradient_estimate = TensorList.zeros_like(policy_architecture.parameters())
        self._num_successes = 0
        if self._use_all_samples:
            batch = instances
            batch_size = len(instances) * self._num_trajectories
        else:
            indices = torch.randint(len(instances), (self._batch_size,))
            batch = [instances[i] for i in indices]
            batch_size = self._batch_size * self._num_trajectories
        self._compute_batch_gradient(fprl, batch, gradient_estimate)
        self._logger.info('END_GRADIENT_COMPUTATION success_rate=%.2f', self._num_successes / batch_size)
        return gradient_estimate

    def _compute_batch_gradient(self, fprl, batch, gradient_estimate):
        instances = itertools.chain(*itertools.repeat(batch, self._num_trajectories))
        input_data = list(map(lambda i: (fprl, i, gradient_estimate), instances))
        results = self._worker_pool.starmap(_run_parallel_trajectories,
                                            input_data)
        env = gp.Env()
        env.setParam(GRB.Param.OutputFlag, 0)
        policy_architecture = fprl.policy_architecture
        for reward, history, instance in results:
            if reward > 0:
                self._num_successes += 1
                gp_model = gp.read(instance, env)
                model = EnhancedModel.from_gurobi_model(gp_model,
                                                        gnn=policy_architecture.gnn,
                                                        convert_ge_cons=True)
                helper = PolicyGradientHelper(fprl, model)
                gradient_estimate.add_to_self(helper.compute_gradient_estimate())
