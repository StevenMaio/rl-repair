"""
An implementation of Evolutionary Strategies in parallel. In this case, the
trajectories are run in parallel. However, I think it may make more sense
to instead run the instances in parallel, that is, the trajectories for a
single instance are run on the same process, but we spawn multiple processes.
"""
import logging
import torch
import itertools
import copy

import torch.multiprocessing as mp

import gurobipy as gp
from gurobipy import GRB

from src.rl.utils import TensorList, NoiseGenerator
from src.utils import create_rng_seeds, get_global_pool
from src.utils.config import NUM_THREADS

from .GradientEstimator import GradientEstimator
from src.rl.mip import EnhancedModel


def _run_trajectory(fprl, instance, noise_seed, rng_seed, noise_std_deviation, mirrored, dropout_p):
    """Runtime procedure for inner loop
    """
    torch.set_num_threads(NUM_THREADS)
    generator = torch.Generator()
    with torch.no_grad():
        policy_architecture_copy = copy.deepcopy(fprl.policy_architecture)
        fprl.policy_architecture = policy_architecture_copy
        env = gp.Env()
        env.setParam(GRB.Param.OutputFlag, 0)
        gp_model = gp.read(instance, env)
        model = EnhancedModel.from_gurobi_model(gp_model,
                                                gnn=policy_architecture_copy.gnn,
                                                convert_ge_cons=True)
        noise_generator = NoiseGenerator(policy_architecture_copy.parameters())
        generator.manual_seed(noise_seed)
        noise = noise_generator.sample(generator=generator,
                                       dropout_p=dropout_p)
        if not mirrored:
            noise.scale(noise_std_deviation)
        else:
            noise.scale(-noise_std_deviation)
        noise.add_to_iterator(policy_architecture_copy.parameters())
        generator.manual_seed(rng_seed)
        fprl.find_solution(model, generator=generator)
        return fprl.reward, noise_seed, mirrored


class EsParallelTrajectories(GradientEstimator):
    _num_successes: int
    _logger: logging.Logger

    # gradient estimator parameters
    _num_trajectories: int
    _batch_size: int
    _use_all_samples: bool
    _noise_std_deviation: float
    _mirrored_sampling: bool
    _dropout_p: float

    _worker_pool: mp.Pool

    def __init__(self,
                 num_trajectories: int,
                 noise_std_deviation: float,
                 batch_size: int = float('inf'),
                 mirrored_sampling=False,
                 dropout_p=0.00):
        self._num_trajectories = num_trajectories
        self._noise_std_deviation = noise_std_deviation
        self._worker_pool = get_global_pool()
        self._num_successes = 0
        self._mirrored_sampling = mirrored_sampling
        self._dropout_p = dropout_p
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
                indices = torch.randint(len(instances), (self._batch_size,))
                batch = [instances[i] for i in indices]
                batch_size = self._batch_size * self._num_trajectories
            gradient_estimate = self._run_instances_in_parallel(batch, fprl)
            self._logger.info('END_GRADIENT_COMPUTATION success_rate=%.2f', self._num_successes / batch_size)
            return gradient_estimate

    def _run_instances_in_parallel(self, instances, fprl):
        policy_architecture = fprl.policy_architecture
        noise_generator = NoiseGenerator(policy_architecture.parameters())
        gradient_estimate = TensorList.zeros_like(policy_architecture.parameters())
        seed_inputs = zip(map(lambda t: t.item(), create_rng_seeds(self._num_trajectories)),
                          map(lambda t: t.item(), create_rng_seeds(self._num_trajectories)))
        if self._mirrored_sampling:
            mirrored_params = [True, False]
        else:
            mirrored_params = [False]
        input_pairs = itertools.product(instances, seed_inputs, mirrored_params)
        input_pairs = [(u, *v, b) for u, v, b in input_pairs]
        results = self._worker_pool.starmap(_run_trajectory,
                                            map(lambda t: (fprl,
                                                           t[0],
                                                           t[1],
                                                           t[2],
                                                           self._noise_std_deviation,
                                                           t[3],
                                                           self._dropout_p),
                                                input_pairs)
                                            )
        self._process_trajectory_results(results, gradient_estimate, noise_generator)
        gradient_estimate.scale(1 / len(instances) / self._num_trajectories / self._noise_std_deviation)
        if self._mirrored_sampling:
            gradient_estimate.scale(0.5)
        return gradient_estimate

    def _process_trajectory_results(self,
                                    results,
                                    gradient_estimate,
                                    noise_generator):
        generator = torch.Generator()
        for reward, noise_seed, mirrored in results:
            if reward > 0:
                generator.manual_seed(noise_seed)
                noise = noise_generator.sample(generator=generator,
                                               dropout_p=self._dropout_p)
                if mirrored:
                    noise.scale(-reward)
                else:
                    noise.scale(reward)
                gradient_estimate.add_to_self(noise)
                self._num_successes += 1
