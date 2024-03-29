"""
Stochastic Optimization with Adaptive Restart (SOAR) is a stochastic optimization
framework introduced in [1]. We recommend  consulting this paper for the details.

Author: Steven Maio

References:
    [1] L. Mathesen, G. Pedrielli, S. H. Ng, and Z. B. Zabinsky, “Stochastic optimization
        with adaptive restart: a framework for integrated local and global learning,"
        J Glob Optim, vol. 79, no. 1, pp. 87–110, Jan. 2021, doi: 10.1007/s10898-020-00937-5.
    [2] SOAR Implementation Note. [[SOAR Implementation#Prelminary Restart Procedure]]. My
        obsidian vault.
"""
from typing import List

from src.soar.restart import RestartMechanism, restart_mechanism_from_config
from src.soar.sampling import ExperimentalDesign, experimental_design_from_config
from src.soar.surrogate import SurrogateModel, surrogate_from_config
from src.soar.termination import TerminationMechanism, termination_mechanism_from_config

from src.mip.heuristic import FixPropRepairLearn

from src.rl.mip import EnhancedModel
from src.rl.utils import DataSet, TensorList
from src.rl.learn.optim import FirstOrderMethod, optimizer_fom_config
from src.rl.learn.gradient import GradientEstimator, gradient_estimator_from_config
from src.rl.learn.trainer import Trainer
from src.utils import get_global_pool, FORMAT_STR, create_rng_seeds
from src.utils.config import *

import logging
import torch
import itertools

import gurobipy as gp
from gurobipy import GRB

DEFAULT_DATA_POINT_INCREMENT = 5  # see [1] for description


def _eval_trajectory(fprl, instance, rng_seed):
    torch.set_num_threads(NUM_THREADS)
    generator = torch.Generator()
    generator.manual_seed(rng_seed)
    with torch.no_grad():
        policy_architecture = fprl.policy_architecture
        env = gp.Env()
        env.setParam(GRB.Param.OutputFlag, 0)
        gp_model = gp.read(instance, env)
        model = EnhancedModel.from_gurobi_model(gp_model,
                                                gnn=policy_architecture.gnn,
                                                convert_ge_cons=True)
        fprl.find_solution(model, generator=generator)
        return fprl.reward


class SOAR(Trainer):
    _logger: logging.getLogger(__package__)

    # local search components
    _optimization_method: FirstOrderMethod
    _gradient_estimator: GradientEstimator

    # local search parameters
    _training_epochs_before_val: int

    # experimental design
    _num_initial_points: int
    _experimental_design: ExperimentalDesign

    _surrogate_model: SurrogateModel
    _termination_mechanism: TerminationMechanism
    _restart_mechanism: RestartMechanism

    # optimization parameters
    _computation_budget: int
    _max_computation_budget: int
    _cross_validation_threshold: float

    _num_trajectories: int
    _eval_in_parallel: bool

    def __init__(self,
                 local_search_method: FirstOrderMethod,
                 gradient_estimator: GradientEstimator,
                 surrogate_model: SurrogateModel,
                 restart_mechanism: RestartMechanism,
                 termination_mechanism: TerminationMechanism,
                 computation_budget: int,
                 experimental_design: ExperimentalDesign,
                 num_initial_points: int,
                 cross_validation_threshold: float,
                 num_eval_trajectories: int = 10,
                 log_file: str = None,
                 eval_in_parallel: bool = False):
        self._logger = logging.getLogger(__package__)

        self._optimization_method = local_search_method
        self._gradient_estimator = gradient_estimator
        self._training_epochs_before_val = 10  # maybe parameterize this at some point

        self._surrogate_model = surrogate_model
        self._restart_mechanism = restart_mechanism
        self._termination_mechanism = termination_mechanism

        self._init_computation_budget = computation_budget

        self._experimental_design = experimental_design
        self._num_initial_points = num_initial_points

        self._cross_validation_threshold = cross_validation_threshold

        self._num_trajectories = num_eval_trajectories
        self._eval_in_parallel = eval_in_parallel
        if eval_in_parallel:
            self._worker_pool = get_global_pool()
        if log_file is not None:
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setFormatter(logging.Formatter(FORMAT_STR))
            self._logger.addHandler(file_handler)

    def train(self,
              fprl: FixPropRepairLearn,
              data_set: DataSet,
              model_output: str = None):
        policy_architecture = fprl.policy_architecture
        parameters = TensorList(policy_architecture.parameters())
        self._computation_budget = self._init_computation_budget
        self._init_and_validate_surrogate_model(fprl, parameters, data_set)
        self._optimization_method.init(fprl)
        while self._computation_budget > 0:
            self._logger.info('BEGIN_GENERAL_ITER remaining_budget=%d',
                              self._computation_budget)
            start_point = self._restart_mechanism.determine_restart_point(self._surrogate_model)
            self._computation_budget -= self._restart_mechanism.get_cost()
            curr_point, obj_val = self._local_search_step(fprl, parameters, data_set, start_point)
            self._surrogate_model.add_point(curr_point, obj_val)
        best_idx = self._surrogate_model.observations.argmax(0)
        self._logger.info('END_TRAINING_BEST_PARAMS best_val_score=%.2f',
                          self._surrogate_model.observations[best_idx])
        best_point = self._surrogate_model.data_points[best_idx]
        parameters.copy_from_1d_tensor(best_point)
        if len(data_set.testing_instances) > 0:
            test_score = self._evaluate_instances(fprl, data_set.testing_instances)
        else:
            test_score = -1
        self._logger.info('END_TRAINING test_score=%.2f', test_score)
        if model_output is not None:
            torch.save(policy_architecture.state_dict(), model_output)

    def _init_and_validate_surrogate_model(self,
                                           fprl: FixPropRepairLearn,
                                           parameters: TensorList,
                                           data_set: DataSet):
        """
        Performs the validation step at the beginning of SOAR.
        :param fprl:
        :param data_set:
        :return:
        """
        num_samples = self._num_initial_points
        self._computation_budget = self._init_computation_budget
        while True:
            initial_points = self._experimental_design.generate_samples(num_samples)
            observations = torch.zeros(num_samples)
            for i in range(num_samples):
                parameters.copy_from_1d_tensor(initial_points[i])
                observations[i] = self._compute_val_score(fprl, data_set)
            self._computation_budget -= self._num_trajectories * num_samples * len(data_set.validation_instances)
            self._surrogate_model.init(initial_points, observations)
            # TODO: do cross validation
            break

    def _local_search_step(self,
                           fprl: FixPropRepairLearn,
                           parameters: TensorList,
                           data_set: DataSet,
                           start_point: torch.Tensor):
        parameters.copy_from_1d_tensor(start_point)
        self._termination_mechanism.init(start_point)
        self._optimization_method.reset()

        continue_local_search = True
        best_val_score = 0.0
        best_point = start_point

        # begin local search
        self._logger.info('BEGIN_LOCAL_SEARCH')
        while continue_local_search and self._computation_budget > 0:
            self._logger.info('LOCAL_SEARCH_INNER_LOOP remaining_budget=%d', self._computation_budget)
            for _ in range(self._training_epochs_before_val):
                gradient_estimate = self._gradient_estimator.estimate_gradient(data_set.training_instances,
                                                                               fprl)
                self._optimization_method.step(fprl.policy_architecture,
                                               gradient_estimate)
            val_score = self._compute_val_score(fprl, data_set)
            if val_score > best_val_score:
                best_point = parameters.flatten()
                best_val_score = val_score
            local_iter_cost = 150  # TODO: replace this a more refined computation

            self._termination_mechanism.update(None, best_val_score, local_iter_cost)
            self._computation_budget -= local_iter_cost
            continue_local_search = not self._termination_mechanism.should_stop()
        return best_point, best_val_score

    @property
    def data_points(self):
        return self._surrogate_model.data_points

    @property
    def observations(self):
        return self._surrogate_model.observations

    def _compute_val_score(self,
                           fprl: FixPropRepairLearn,
                           data_set: DataSet):
        val_score = 0
        if len(data_set.validation_instances) > 0:
            val_score = self._evaluate_instances(fprl, data_set.validation_instances)
            self._logger.info('VAL_COMPUTATION val_score=%.2f',
                              val_score)
        return val_score

    def _evaluate_instances(self,
                            fprl: FixPropRepairLearn,
                            instances: List[str]):
        if self._eval_in_parallel:
            return self._evaluate_instances_parallel(fprl, instances)
        else:
            return self._evaluate_instances_serial(fprl, instances)

    def _evaluate_instances_parallel(self, fprl, instances):
        batch_size = len(instances) * self._num_trajectories
        pool_inputs = [itertools.repeat(i, self._num_trajectories) for i in instances]
        pool_inputs = zip(itertools.chain(*pool_inputs),
                          map(lambda t: t.item(), create_rng_seeds(batch_size)))
        results = self._worker_pool.starmap(_eval_trajectory,
                                            map(lambda t: (fprl,
                                                           t[0],
                                                           t[1]),
                                                pool_inputs)
                                            )
        num_successes = 0
        for r in results:
            if r > 0:
                num_successes += 1
        return num_successes / batch_size

    def _evaluate_instances_serial(self, fprl, instances):
        policy_architecture = fprl.policy_architecture
        num_successes = 0
        batch_size = len(instances) * self._num_trajectories
        for instance in instances:
            env = gp.Env()
            env.setParam(GRB.Param.OutputFlag, 0)
            gp_model = gp.read(instance, env)
            model = EnhancedModel.from_gurobi_model(gp_model,
                                                    gnn=policy_architecture.gnn,
                                                    convert_ge_cons=True)
            for trajectory_num in range(self._num_trajectories):
                fprl.find_solution(model)
                if fprl.reward != 0:
                    num_successes += 1
                if trajectory_num < self._num_trajectories - 1:
                    model.reset()
        return num_successes / batch_size

    @staticmethod
    def from_config(config: dict):
        params = config[PARAMS]
        gradient_estimator = gradient_estimator_from_config(config[GRADIENT_ESTIMATOR])
        optimization_method = optimizer_fom_config(config[OPTIMIZATION_METHOD])

        support = torch.load(config[SUPPORT])
        corr_parameters = torch.load(config[CORRELATION_PARAMETERS])

        surrogate_config = config[SURROGATE_MODEL]
        surrogate_config[PARAMS].update({
            SUPPORT: support,
            CORRELATION_PARAMETERS: corr_parameters
        })
        surrogate_model = surrogate_from_config(surrogate_config)

        restart_mechanism = restart_mechanism_from_config(config[RESTART_MECHANISM])
        termination_mechanism = termination_mechanism_from_config(config[TERMINATION_MECHANISM])

        experimental_design_config = config[EXPERIMENTAL_DESIGN]
        experimental_design_config[PARAMS].update({
            SUPPORT: support
        })
        experimental_design = experimental_design_from_config(experimental_design_config)

        params.update({
            LOCAL_SEARCH_METHOD: optimization_method,
            GRADIENT_ESTIMATOR: gradient_estimator,
            SURROGATE_MODEL: surrogate_model,
            RESTART_MECHANISM: restart_mechanism,
            TERMINATION_MECHANISM: termination_mechanism,
            EXPERIMENTAL_DESIGN: experimental_design
        })
        soar = SOAR(**params)
        return soar
