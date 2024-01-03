"""
A modification of FirstOrderTrainer that uses a restart strategy outlined in [1] (see
§3.4).

References:
    [1] D. Wierstra, T. Schaul, T. Glasmachers, Y. Sun, and J. Schmidhuber,
        “Natural Evolution Strategies.” arXiv, Jun. 22, 2011. Accessed: Oct. 18, 2023.
        [Online]. Available: http://arxiv.org/abs/1106.4487

"""
import itertools

import torch
import torch.multiprocessing as mp

from src.utils.config import NUM_THREADS, PARAMS, GRADIENT_ESTIMATOR, OPTIMIZATION_METHOD, \
    VAL_PROGRESS_CHECKER
from src.rl.learn.gradient import gradient_estimator_from_config, GradientEstimator
from src.rl.learn.optim import FirstOrderMethod, optimizer_fom_config
from src.rl.learn.val import progress_checker_from_config

from src.mip.heuristic import FixPropRepairLearn

from src.rl.utils import DataSet

import logging

from src.rl.learn.val import ValidationProgressChecker
from src.rl.architecture import PolicyArchitecture
from src.rl.mip import EnhancedModel
from src.rl.params import GnnParams

import gurobipy as gp
from gurobipy import GRB

from src.utils import FORMAT_STR, get_global_pool, create_rng_seeds


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


class FirstOrderRestartTrainer:
    _optimization_method: FirstOrderMethod
    _gradient_estimator: GradientEstimator
    _num_epochs: int
    _logger: logging.Logger
    _iters_to_progress_check: int
    _num_trajectories: int
    _val_progress_checker: ValidationProgressChecker
    _minimum_training_epochs: int
    _remaining_budget_allocation: float

    # parallel stuff
    _eval_in_parallel: bool
    _worker_pool: mp.Pool

    _best_val_score: float
    _init_test_score: float
    _final_test_score: float
    _current_epoch: int
    _best_restart_policy: PolicyArchitecture

    # restart results
    _best_restart_score: float
    _best_restart_policy: PolicyArchitecture

    def __init__(self,
                 optimization_method: FirstOrderMethod,
                 gradient_estimator: GradientEstimator,
                 num_epochs: int,
                 iters_to_progress_check: int,
                 val_progress_checker: ValidationProgressChecker,
                 minimum_restart_epochs: int,
                 remaining_budget_allocation: float,
                 num_eval_trajectories: int = 5,
                 log_file: str = None,
                 eval_in_parallel: bool = False):
        self._optimization_method = optimization_method
        self._gradient_estimator = gradient_estimator
        self._num_epochs = num_epochs
        self._iters_to_progress_check = iters_to_progress_check
        self._num_trajectories = num_eval_trajectories
        self._logger = logging.getLogger(__package__)
        self._best_training_policy = PolicyArchitecture(GnnParams)
        self._best_restart_policy = PolicyArchitecture(GnnParams)
        self._val_progress_checker = val_progress_checker
        self._eval_in_parallel = eval_in_parallel
        self._minimum_training_epochs = minimum_restart_epochs
        self._remaining_budget_allocation = remaining_budget_allocation
        if self._eval_in_parallel:
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
        self._optimization_method.init(fprl)
        num_restarts = self._compute_num_restarts(self._num_epochs,
                                                  self._remaining_budget_allocation,
                                                  self._minimum_training_epochs)
        self._logger.info('BEGIN_INNER_RNG_SEED rng_seed=%d',
                          torch.initial_seed())
        if len(data_set.testing_instances) > 0:
            init_test_score = self._evaluate_instances(fprl, data_set.testing_instances)
        else:
            init_test_score = -1
        self._logger.info('BEGIN_TRAINING_TEST_SCORE test_score=%.2f',
                          init_test_score)
        best_val_score = -float('inf')
        for restart_num in range(num_restarts):
            num_training_epochs = self._compute_num_training_epochs(self._num_epochs,
                                                                    self._remaining_budget_allocation,
                                                                    restart_num)
            # sample new starting point (except on the first iteration)
            if restart_num > 0:
                for p in policy_architecture.parameters():
                    torch.nn.init.uniform_(p, a=-1.0, b=1.0)
            val_score = self._train_restart_loop(fprl,
                                                 data_set,
                                                 num_training_epochs,
                                                 restart_num)
            if val_score > best_val_score:
                best_val_score = val_score
                self._best_training_policy.load_state_dict(policy_architecture.state_dict())
                if model_output is not None:
                    torch.save(self._best_training_policy.state_dict(), model_output)
        policy_architecture.load_state_dict(self._best_training_policy.state_dict())
        if len(data_set.testing_instances) > 0:
            test_score = self._evaluate_instances(fprl, data_set.testing_instances)
        else:
            test_score = -1
        self._logger.info('END_TRAINING best_val_score=%.2f test_score=%.2f',
                          best_val_score,
                          test_score)

    def _train_restart_loop(self,
                            fprl: FixPropRepairLearn,
                            data_set: DataSet,
                            num_epochs: int,
                            restart_num: int):
        policy_architecture = fprl.policy_architecture
        self._logger.info('BEGIN_INNER_TRAINING restart_num=%d',
                          restart_num)
        self._optimization_method.reset()
        self._best_val_score = 0
        self._val_progress_checker.reset()
        # truncates num of epochs to the quotient of the num_epochs and iters_to_progress_check
        num_epochs = self._iters_to_progress_check * (num_epochs // self._iters_to_progress_check)
        for epoch in range(num_epochs):
            gradient_estimate = self._gradient_estimator.estimate_gradient(data_set.training_instances,
                                                                           fprl)
            self._optimization_method.step(fprl.policy_architecture,
                                           gradient_estimate)
            self._logger.info('END_OF_EPOCH restart_num=%d epoch=%d best_val=%.2f',
                              restart_num,
                              epoch,
                              self._best_val_score)
            if (epoch + 1) % self._iters_to_progress_check == 0:
                self._check_progress(fprl, data_set)
        # load best policy found during inner loop
        policy_architecture.load_state_dict(self._best_restart_policy.state_dict())
        self._logger.info('END_INNER_TRAINING restart_num=%d best_val_score=%.2f',
                          restart_num,
                          self._best_val_score)
        return self._best_val_score

    def _check_progress(self, fprl, data_set):
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
                self._best_restart_policy.load_state_dict(policy_architecture.state_dict())
            if not self._val_progress_checker.continue_training():
                policy_architecture.load_state_dict(self._best_restart_policy.state_dict())
                self._optimization_method.reset()
                self._val_progress_checker.reset()
                self._logger.info('PARAMETER_RESET')

    def _evaluate_instances(self, fprl, instances):
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
    def _compute_num_restarts(total_budget, p, minimum_budget):
        """
        Computation comes from analysis of restart strategy outlined in §3.4 of [1].
        The number of restarts $N$ is given by
        $$
        N = \lfloor(\log(pT) - \log(\epsilon))/\log(1/(1-p))\rfloor - 1.
        $$
        :param total_budget: $T$
        :param p: $p\in(0, 1)$
        :param minimum_budget: $\epsilon > 0$
        :return:
        """
        num_restarts = torch.ceil(
            (torch.log(torch.Tensor([total_budget * p]))
             - torch.log(torch.Tensor([minimum_budget]))) / torch.log(torch.Tensor([1 / (1 - p)]))
        ) - 1
        return int(num_restarts.item())

    @staticmethod
    def _compute_num_training_epochs(total_budget, p, restart_num):
        """
        Computes the number of training epochs in the current restart iteration
        based on the restart strategy outlined in §3.4 of [1].
        :param total_budget:
        :param p:
        :param restart_num:
        :return:
        """
        num_epochs = torch.floor(torch.Tensor([total_budget * p
                                 * torch.pow(torch.Tensor([1 - p]),
                                             torch.Tensor([restart_num]))
                                               ]))
        return int(num_epochs.item())

    @staticmethod
    def from_config(config: dict):
        params = config[PARAMS]
        gradient_estimator = gradient_estimator_from_config(config[GRADIENT_ESTIMATOR])
        optimization_method = optimizer_fom_config(config[OPTIMIZATION_METHOD])
        # TODO: create the validation progress checker
        val_progress_checker = progress_checker_from_config(config[VAL_PROGRESS_CHECKER])

        return FirstOrderRestartTrainer(optimization_method=optimization_method,
                                        gradient_estimator=gradient_estimator,
                                        val_progress_checker=val_progress_checker,
                                        **params)
