import torch
import os

from src.rl.utils.DataSet import DataSet

from src.mip.heuristic import FixPropRepairLearn
from src.mip.heuristic.repair import LearnableRepairWalk
from src.mip.params import RepairWalkParams
from src.mip.propagation import LinearConstraintPropagator

from src.rl.architecture import PolicyArchitecture
from src.rl.params import GnnParams
from src.rl.learn import EvolutionaryStrategiesSerial, GradientAscent, EsParallelTrajectories, \
    EsParallelInstances, Adam, FoValTrainer

from src.utils import initialize_logger

import logging

from src.utils.config import *

import torch.multiprocessing as mp


def serial_es_main():
    instance_dir = os.sep.join([PROJECT_ROOT, INSTANCES])
    instances = [os.sep.join([instance_dir, f]) for f in os.listdir(instance_dir) if '.opb' in f or '.mps' in f]
    data_set = DataSet(instances,
                       validation_portion=VAL_PORTION,
                       testing_portion=TEST_PORTION,
                       rng_seed=DATA_SPLIT_SEED)
    input_policy = os.sep.join([PROJECT_ROOT, INPUT_MODEL])

    # create and load policy architecture
    policy_architecture = PolicyArchitecture(GnnParams)
    policy_architecture.load_state_dict(torch.load(input_policy))
    repair_strat = LearnableRepairWalk(RepairWalkParams(),
                                       policy_architecture.cons_scoring_function,
                                       policy_architecture.var_scoring_function,
                                       sample_indices=SAMPLE_INDICES)
    fprl = FixPropRepairLearn(policy_architecture.fixing_order_architecture,
                              policy_architecture.value_fixing_architecture,
                              repair_strat,
                              LinearConstraintPropagator(),
                              policy_architecture,
                              sample_indices=SAMPLE_INDICES,
                              in_training=True,
                              discount_factor=DISCOUNT_FACTOR,
                              max_backtracks=MAX_BACKTRACKS)

    gradient_estimator = EvolutionaryStrategiesSerial(num_trajectories=NUM_TRAJECTORIES,
                                                      learning_parameter=LEARNING_PARAMETER,
                                                      batch_size=BATCH_SIZE)
    optimization_method = Adam(fprl=fprl,
                               step_size=LEARNING_RATE,
                               first_moment_decay_rate=FIRST_MOMENT_DECAY_RATE,
                               second_moment_decay_rate=SECOND_MOMENT_DECAY_RATE,
                               epsilon=EPSILON)
    trainer = FoValTrainer(optimization_method=optimization_method,
                           num_epochs=NUM_EPOCHS,
                           gradient_estimator=gradient_estimator,
                           iters_to_val=ITERS_TO_VAL,
                           num_allowable_worse_vals=NUM_ALLOWABLE_WORSE_VALS,
                           num_trajectories=NUM_VAL_TRAJECTORIES,
                           log_file=TRAINING_LOG)
    trainer.train(fprl=fprl,
                  data_set=data_set,
                  save_rate=SAVE_RATE,
                  model_output=OUTPUT_MODEL)


def parallel_instances_es_main():
    instance_dir = os.sep.join([PROJECT_ROOT, INSTANCES])
    instances = [os.sep.join([instance_dir, f]) for f in os.listdir(instance_dir) if '.opb' in f or '.mps' in f]
    data_set = DataSet(instances,
                       validation_portion=VAL_PORTION,
                       testing_portion=TEST_PORTION,
                       rng_seed=DATA_SPLIT_SEED)
    input_policy = os.sep.join([PROJECT_ROOT, INPUT_MODEL])

    # create and load policy architecture
    policy_architecture = PolicyArchitecture(GnnParams)
    policy_architecture.load_state_dict(torch.load(input_policy))
    repair_strat = LearnableRepairWalk(RepairWalkParams(),
                                       policy_architecture.cons_scoring_function,
                                       policy_architecture.var_scoring_function,
                                       sample_indices=SAMPLE_INDICES)
    fprl = FixPropRepairLearn(policy_architecture.fixing_order_architecture,
                              policy_architecture.value_fixing_architecture,
                              repair_strat,
                              LinearConstraintPropagator(),
                              policy_architecture,
                              sample_indices=SAMPLE_INDICES,
                              in_training=True,
                              discount_factor=DISCOUNT_FACTOR,
                              max_backtracks=MAX_BACKTRACKS)

    gradient_estimator = EsParallelInstances(num_trajectories=NUM_TRAJECTORIES,
                                             learning_parameter=LEARNING_PARAMETER,
                                             num_workers=NUM_WORKERS,
                                             batch_size=BATCH_SIZE)
    optimization_method = Adam(fprl=fprl,
                               step_size=LEARNING_RATE,
                               first_moment_decay_rate=FIRST_MOMENT_DECAY_RATE,
                               second_moment_decay_rate=SECOND_MOMENT_DECAY_RATE,
                               epsilon=EPSILON)
    trainer = FoValTrainer(optimization_method=optimization_method,
                           num_epochs=NUM_EPOCHS,
                           gradient_estimator=gradient_estimator,
                           iters_to_val=ITERS_TO_VAL,
                           num_allowable_worse_vals=NUM_ALLOWABLE_WORSE_VALS,
                           num_trajectories=NUM_VAL_TRAJECTORIES,
                           log_file=TRAINING_LOG)
    trainer.train(fprl=fprl,
                  data_set=data_set,
                  save_rate=SAVE_RATE,
                  model_output=OUTPUT_MODEL)


def parallel_trajectories_es_main():
    instance_dir = os.sep.join([PROJECT_ROOT, INSTANCES])
    instances = [os.sep.join([instance_dir, f]) for f in os.listdir(instance_dir) if '.opb' in f or '.mps' in f]
    data_set = DataSet(instances,
                       validation_portion=VAL_PORTION,
                       testing_portion=TEST_PORTION,
                       rng_seed=DATA_SPLIT_SEED)
    input_policy = os.sep.join([PROJECT_ROOT, INPUT_MODEL])

    # create and load policy architecture
    policy_architecture = PolicyArchitecture(GnnParams)
    policy_architecture.load_state_dict(torch.load(input_policy))
    repair_strat = LearnableRepairWalk(RepairWalkParams(),
                                       policy_architecture.cons_scoring_function,
                                       policy_architecture.var_scoring_function,
                                       sample_indices=SAMPLE_INDICES)
    fprl = FixPropRepairLearn(policy_architecture.fixing_order_architecture,
                              policy_architecture.value_fixing_architecture,
                              repair_strat,
                              LinearConstraintPropagator(),
                              policy_architecture,
                              sample_indices=SAMPLE_INDICES,
                              in_training=True,
                              discount_factor=DISCOUNT_FACTOR,
                              max_backtracks=MAX_BACKTRACKS)

    gradient_estimator = EsParallelTrajectories(num_trajectories=NUM_TRAJECTORIES,
                                                learning_parameter=LEARNING_PARAMETER,
                                                num_workers=NUM_WORKERS,
                                                batch_size=BATCH_SIZE)
    optimization_method = GradientAscent(learning_rate=LEARNING_RATE)
    trainer = FoValTrainer(optimization_method=optimization_method,
                           num_epochs=NUM_EPOCHS,
                           gradient_estimator=gradient_estimator,
                           iters_to_val=ITERS_TO_VAL,
                           num_allowable_worse_vals=NUM_ALLOWABLE_WORSE_VALS,
                           num_trajectories=NUM_VAL_TRAJECTORIES,
                           log_file=TRAINING_LOG)
    trainer.train(fprl=fprl,
                  data_set=data_set,
                  save_rate=SAVE_RATE,
                  model_output=OUTPUT_MODEL)


if __name__ == '__main__':
    initialize_logger(level=logging.INFO)
    mp.set_start_method('forkserver')
    import time

    # start = time.time()
    # parallel_instances_es_main()
    # parallel_time = time.time() - start

    start = time.time()
    serial_es_main()
    serial_time = time.time() - start

    print(f'serial {serial_time}')
    # print(f'parallel {parallel_time}')
