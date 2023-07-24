"""
An implementation of Evolutionary Strategies in parallel. In this case, the
trajectories are run in parallel. However, I think it may make more sense
to instead run the instances in parallel, that is, the trajectories for a
single instance are run on the same process, but we spawn multiple processes.
"""
import random
import logging
import torch

import itertools

import torch.multiprocessing as mp

import gurobipy as gp
from gurobipy import GRB

from src.rl.utils import TensorList, NoiseGenerator
from src.utils import create_rng_seeds

from .GradientEstimator import GradientEstimator
from src.rl.mip import EnhancedModel


def _run_instance(fprl, instance, rng_seeds, learning_param):
    """Runtime procedure for inner loop
    """
    with torch.no_grad():
        policy_architecture = fprl.policy_architecture
        env = gp.Env()
        env.setParam(GRB.Param.OutputFlag, 0)
        gp_model = gp.read(instance, env)
        model = EnhancedModel.from_gurobi_model(gp_model,
                                                gnn=policy_architecture.gnn,
                                                convert_ge_cons=True)
        noise_generator = NoiseGenerator(policy_architecture.parameters())
        num_trajectories = len(rng_seeds)
        results = []
        for iter_num, seed in enumerate(rng_seeds):
            torch.manual_seed(seed)
            noise = noise_generator.sample()
            noise.scale(learning_param)
            noise.add_to_iterator(policy_architecture.parameters())
            fprl.find_solution(model)
            noise.scale(-1)
            noise.add_to_iterator(policy_architecture.parameters())
            results.append((fprl.reward, seed))
            if iter_num != num_trajectories - 1:
                model.reset()
        return results


class EsParallelInstances(GradientEstimator):
    _num_successes: int
    _logger: logging.Logger

    # gradient estimator parameters
    _num_trajectories: int
    _batch_size: int
    _use_all_samples: bool
    _learning_parameter: float
    _num_workers: int

    _worker_pool: mp.Pool

    def __init__(self,
                 num_trajectories: int,
                 num_workers: int,
                 learning_parameter: float,
                 batch_size: int = float('inf')):
        self._num_trajectories = num_trajectories
        self._learning_parameter = learning_parameter
        self._num_workers = num_workers
        self._worker_pool = mp.Pool(num_workers)
        self._num_successes = 0
        if batch_size == float('inf'):
            self._use_all_samples = True
        else:
            self._use_all_samples = False
        self._batch_size = batch_size
        self._logger = logging.getLogger(__package__)

    def estimate_gradient(self, instances, fprl):
        with torch.no_grad():
            self._num_successes = 0
            if self._use_all_samples:
                batch = instances
                batch_size = len(instances) * self._num_trajectories
            else:
                batch = random.choices(instances, k=self._batch_size)
                batch_size = self._batch_size * self._num_trajectories
            gradient_estimate = self._run_instances_in_parallel(batch, fprl)
            self._logger.info('success_rate=%.2f', self._num_successes / batch_size)
            return gradient_estimate

    def _run_instances_in_parallel(self, instances, fprl):
        policy_architecture = fprl.policy_architecture
        noise_generator = NoiseGenerator(policy_architecture.parameters())
        gradient_estimate = TensorList.zeros_like(policy_architecture.parameters())
        results = self._worker_pool.starmap(_run_instance,
                                            map(lambda p: (fprl,
                                                           p,
                                                           create_rng_seeds(self._num_trajectories),
                                                           self._learning_parameter,
                                                           ),
                                                instances))
        self._process_trajectory_results(results, gradient_estimate, noise_generator)
        gradient_estimate.scale(1 / len(instances) / self._num_trajectories / self._learning_parameter)
        return gradient_estimate

    def _process_trajectory_results(self,
                                    results,
                                    gradient_estimate,
                                    noise_generator):
        for reward, rng_seed in itertools.chain(*results):
            if reward > 0:
                torch.manual_seed(rng_seed)
                noise = noise_generator.sample()
                gradient_estimate.add_to_self(noise)
                self._num_successes += 1
