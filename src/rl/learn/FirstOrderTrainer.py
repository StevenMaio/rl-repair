import torch

from .FirstOrderMethod import FirstOrderMethod
from .GradientEstimator import GradientEstimator

from src.mip.heuristic import FixPropRepairLearn

from src.rl.utils import DataSet

import logging

from .val import ValidationProgressChecker
from ..architecture import PolicyArchitecture
from ..mip import EnhancedModel
from ..params import GnnParams

import gurobipy as gp
from gurobipy import GRB

from src.utils import FORMAT_STR


class FoValTrainer:
    _optimization_method: FirstOrderMethod
    _gradient_estimator: GradientEstimator
    _num_epochs: int
    _logger: logging.Logger
    _iters_to_progress_check: int
    _num_allowable_worse_vals: int
    _num_trajectories: int
    _val_progress_checker: ValidationProgressChecker

    #
    _best_val_score: float
    _init_test_score: float
    _final_test_score: float
    _current_epoch: int
    _best_policy: PolicyArchitecture

    def __init__(self,
                 optimization_method: FirstOrderMethod,
                 gradient_estimator: GradientEstimator,
                 num_epochs: int,
                 iters_to_progress_check: int,
                 val_progress_checker: ValidationProgressChecker,
                 num_allowable_worse_vals: int = 5,
                 num_trajectories: int = 5,
                 log_file: str = None):
        self._optimization_method = optimization_method
        self._gradient_estimator = gradient_estimator
        self._num_epochs = num_epochs
        self._iters_to_progress_check = iters_to_progress_check
        self._num_allowable_worse_vals = num_allowable_worse_vals
        self._num_trajectories = num_trajectories
        self._logger = logging.getLogger(__package__)
        self._best_policy = PolicyArchitecture(GnnParams)
        self._val_progress_checker = val_progress_checker
        if log_file is not None:
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setFormatter(logging.Formatter(FORMAT_STR))
            self._logger.addHandler(file_handler)

    def train(self,
              fprl: FixPropRepairLearn,
              data_set: DataSet,
              model_output: str = None,
              restart: bool = True,
              trainer_data: str = None):
        policy_architecture = fprl.policy_architecture
        if restart:
            self._optimization_method.reset()
            self._best_policy.load_state_dict(policy_architecture.state_dict())
            self._current_epoch = 0
            self._best_val_score = 0
            self._val_progress_checker.reset()
            if len(data_set.testing_instances) > 0:
                self._init_test_score = self._evaluate_instances(fprl, data_set.testing_instances)
            else:
                self._init_test_score = -1
            self._logger.info('BEGIN_TRAINING test_score=%.2f', self._init_test_score)
        for epoch in range(self._current_epoch, self._num_epochs):
            gradient_estimate = self._gradient_estimator.estimate_gradient(data_set.training_instances,
                                                                           fprl)
            self._optimization_method.step(fprl.policy_architecture,
                                           gradient_estimate)
            self._logger.info('END_OF_EPOCH epoch=%d best_val=%.2f', epoch, self._best_val_score)
            if (epoch + 1) % self._iters_to_progress_check == 0:
                self._check_progress(fprl, data_set, model_output, trainer_data)
        # load best policy architecture
        policy_architecture.load_state_dict(self._best_policy.state_dict())
        if model_output is not None:
            torch.save(self._best_policy.state_dict(), model_output)
        if len(data_set.testing_instances) > 0:
            test_score = self._evaluate_instances(fprl, data_set.testing_instances)
        else:
            test_score = -1
        self._logger.info('END_TRAINING test_score=%.2f', test_score)

    def _check_progress(self, fprl, data_set, model_output, trainer_data):
        policy_architecture = fprl.policy_architecture
        if len(data_set.validation_instances) > 0:
            raw_val_score = self._evaluate_instances(fprl, data_set.validation_instances)
            self._val_progress_checker.update_progress(raw_val_score)
            val_score = self._val_progress_checker.corrected_score()
            self._logger.info('VAL_COMPUTATION raw_val_score=%.2f val_score=%.2f',
                              raw_val_score,
                              val_score)
            if val_score > self._best_val_score:
                self._best_val_score = val_score
                self._best_policy.load_state_dict(policy_architecture.state_dict())
            if not self._val_progress_checker.continue_training():
                policy_architecture.load_state_dict(self._best_policy.state_dict())
                self._optimization_method.reset()
                self._val_progress_checker.reset()
                self._logger.info('PARAMETER_RESET')
        if model_output is not None:
            torch.save(self._best_policy.state_dict(), model_output)
        # TODO: save trainer data

    def _evaluate_instances(self, fprl, val_data):
        policy_architecture = fprl.policy_architecture
        num_successes = 0
        batch_size = len(val_data) * self._num_trajectories
        for instance in val_data:
            # TODO: allow for this to be done in parallel
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
