import logging
import torch

import gurobipy as gp
from gurobipy import GRB

from src.rl.utils import TensorList, NoiseGenerator
from src.rl.learn.gradient.GradientEstimator import GradientEstimator
from src.rl.mip import EnhancedModel


class EvolutionaryStrategiesSerial(GradientEstimator):
    _num_successes: int

    # gradient estimator parameters
    _num_trajectories: int
    _batch_size: int
    _use_all_samples: bool
    _noise_std_deviation: float
    _logger: logging.Logger

    def __init__(self,
                 num_trajectories: int,
                 noise_std_deviation: float,
                 batch_size: int = float('inf')):
        self._num_trajectories = num_trajectories
        self._noise_std_deviation = noise_std_deviation
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
                gradient_estimate = self._estimate_gradient_use_all_samples(instances, fprl)
                batch_size = len(instances) * self._num_trajectories
            else:
                gradient_estimate = self._estimate_gradient_batched_iteration(instances, fprl)
                batch_size = self._batch_size * self._num_trajectories
            self._logger.info('END_GRADIENT_COMPUTATION success_rate=%.2f', self._num_successes / batch_size)
            return gradient_estimate

    def _estimate_gradient_use_all_samples(self, instances, fprl):
        policy_architecture = fprl.policy_architecture
        noise_generator = NoiseGenerator(policy_architecture.parameters())
        gradient_estimate = TensorList.zeros_like(policy_architecture.parameters())
        for problem_instance in instances:
            gradient_estimate.add_to_self(self._get_instance_gradient_estimate(fprl,
                                                                               problem_instance,
                                                                               noise_generator))
        gradient_estimate.scale(1 / len(instances) / self._noise_std_deviation)
        return gradient_estimate

    def _estimate_gradient_batched_iteration(self, instances, fprl):
        policy_architecture = fprl.policy_architecture
        noise_generator = NoiseGenerator(policy_architecture.parameters())
        gradient_estimate = TensorList.zeros_like(policy_architecture.parameters())
        indices = torch.randint(len(instances), (self._batch_size,))
        batch = [instances[i] for i in indices]
        for problem_instance in batch:
            gradient_estimate.add_to_self(self._get_instance_gradient_estimate(fprl,
                                                                               problem_instance,
                                                                               noise_generator))
        gradient_estimate.scale(1 / self._batch_size / self._noise_std_deviation)
        return gradient_estimate

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
            noise.scale(self._noise_std_deviation)
            noise.add_to_iterator(policy_architecture.parameters())
            fprl.find_solution(model)
            noise.scale(-1)
            noise.add_to_iterator(policy_architecture.parameters())
            if fprl.reward != 0:
                noise.scale(-fprl.reward / self._noise_std_deviation)
                gradient_estimate.add_to_self(noise)
                self._num_successes += 1
            model.reset()
        gradient_estimate.scale(1 / self._num_trajectories)
        return gradient_estimate
