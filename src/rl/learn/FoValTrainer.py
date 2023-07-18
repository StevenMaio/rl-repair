import torch

from .FirstOrderMethod import FirstOrderMethod
from .GradientEstimator import GradientEstimator

from src.mip.heuristic import FixPropRepairLearn

from src.rl.utils import DataSet

import logging

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
    _iters_to_val: int
    _num_allowable_worse_vals: int
    _num_trajectories: int

    def __init__(self,
                 optimization_method: FirstOrderMethod,
                 gradient_estimator: GradientEstimator,
                 num_epochs: int,
                 iters_to_val: int,
                 num_allowable_worse_vals: int = 5,
                 num_trajectories: int = 5,
                 log_file: str = None):
        self._optimization_method = optimization_method
        self._gradient_estimator = gradient_estimator
        self._num_epochs = num_epochs
        self._iters_to_val = iters_to_val
        self._num_allowable_worse_vals = num_allowable_worse_vals
        self._num_trajectories = num_trajectories
        self._logger = logging.getLogger(__package__)
        if log_file is not None:
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setFormatter(logging.Formatter(FORMAT_STR))
            self._logger.addHandler(file_handler)

    def train(self,
              fprl: FixPropRepairLearn,
              data_set: DataSet,
              save_rate: int = float('inf'),
              model_output: str = None):
        self._optimization_method.reset()
        policy_architecture = fprl.policy_architecture
        # TODO: init training -- save current model, and av
        best_architecture = PolicyArchitecture(GnnParams)
        best_architecture.load_state_dict(policy_architecture.state_dict())
        best_val_score = self._compute_val_score(fprl, data_set.validation_instances)
        num_worse_val = 0
        self._logger.info('begin training -- val_score=%.2f', best_val_score)
        for epoch in range(self._num_epochs):
            gradient_estimate = self._gradient_estimator.estimate_gradient(data_set.training_instances,
                                                                           fprl)
            self._optimization_method.step(fprl.policy_architecture,
                                           gradient_estimate)
            # save model at epoch intervals
            if model_output is not None and (epoch + 1) % save_rate == 0:
                torch.save(policy_architecture.state_dict(), model_output)
            if (epoch + 1) % self._iters_to_val == 0:
                val_score = self._compute_val_score(fprl, data_set.validation_data)
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_architecture.load_state_dict(policy_architecture.state_dict())
                    num_worse_val = 0
                    self._logger.info('validation_score improvement -- val_score=%.2f', best_val_score)
                else:
                    num_worse_val += 1
                    if num_worse_val == self._num_allowable_worse_vals:
                        policy_architecture.load_state_dict(best_architecture.state_dict())
                        num_worse_val = 0
                        self._optimization_method.reset()
                        self._logger.info('resetting architecture')
            self._logger.info('end of epoch %d. best_val=%.2f', epoch, best_val_score)
        # save model at end
        if model_output is not None:
            torch.save(policy_architecture.state_dict(), model_output)

    def _compute_val_score(self, fprl, val_data):
        prev_sample_indices = fprl.sample_indices
        policy_architecture = fprl.policy_architecture
        fprl.sample_indices = False
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
        fprl.sample_indices = prev_sample_indices
        return num_successes / batch_size
