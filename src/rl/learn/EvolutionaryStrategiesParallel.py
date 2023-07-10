"""
An implementation of Evolutionary Strategies in parallel. In this case, the
trajectories are run in parallel. However, I think it may make more sense
to instead run the instances in parallel, that is, the trajectories for a
single instance are run on the same process, but we spawn multiple processes.
"""
import random
import logging
import torch

import torch.multiprocessing as mp

import gurobipy as gp
from gurobipy import GRB

from src.rl.utils import TensorList, NoiseGenerator
from .GradientEstimator import GradientEstimator
from src.rl.mip import EnhancedModel


def run_trajectory(data):
    """Runtime procedure for inner loop
    """
    with torch.no_grad():
        instance, fprl, noise, learning_param, state_dict = data
        # TODO: copy the FPRL architecture
        # TODO: add perturbation to copied architecture
        # TODO: load the given instance
        # TODO: run FPRL on the instance
        # TODO: return the reward from running FPRL
        policy_architecture = fprl.policy_architecture
        noise.scale(learning_param)
        noise.add_to_iterator(policy_architecture.parameters())
        env = gp.Env()
        env.setParam(GRB.Param.OutputFlag, 0)
        gp_model = gp.read(instance, env)
        model = EnhancedModel.from_gurobi_model(gp_model,
                                                gnn=policy_architecture.gnn,
                                                convert_ge_cons=True)
        fprl.find_solution(model)
        return fprl.reward


class EvolutionaryStrategiesParallel(GradientEstimator):
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
        state_dict = policy_architecture.state_dict()
        for problem_instance in instances:
            perturbations = [noise_generator.sample(in_shared_mem=False) for _ in range(self._num_trajectories)]
            results = self._worker_pool.map(run_trajectory,
                                            map(lambda p: (problem_instance,
                                                           fprl,
                                                           p,
                                                           self._learning_parameter,
                                                           state_dict),
                                                perturbations))
            self._process_trajectory_results(results, perturbations, gradient_estimate)
        gradient_estimate.scale(1 / len(instances) / self._num_trajectories / self._learning_parameter)
        return gradient_estimate

    def _process_trajectory_results(self,
                                    results,
                                    perturbations,
                                    gradient_estimate):
        for reward, noise in zip(results, perturbations):
            if reward > 0:
                noise.scale(reward)
                gradient_estimate.add_to_self(noise)
                self._num_successes += 1

    def _get_instance_gradient_estimate(self, fprl, instance, noise_generator):
        policy_architecture = fprl.policy_architecture
        gradient_estimate = TensorList.zeros_like(policy_architecture.parameters())
        env = gp.Env()
        env.setParam(GRB.Param.OutputFlag, 0)
        gp_model = gp.read(instance, env)
        model = EnhancedModel.from_gurobi_model(gp_model,
                                                gnn=policy_architecture.gnn,
                                                convert_ge_cons=True)
        for trajectory_num in range(self._num_trajectories):
            noise = noise_generator.sample()
            noise.scale(self._learning_parameter)
            noise.add_to_iterator(policy_architecture.parameters())
            fprl.find_solution(model)
            noise.scale(-1)
            noise.add_to_iterator(policy_architecture.parameters())
            if fprl.reward != 0:
                noise.scale(fprl.reward)
                gradient_estimate.add_to_self(noise)
                self._num_successes += 1
            model.reset()
        gradient_estimate.scale(1 / self._num_trajectories)
        return gradient_estimate
