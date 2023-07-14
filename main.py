import torch
import os

from src.mip.heuristic import FixPropRepairLearn, FixPropRepair
from src.mip.heuristic.repair import LearnableRepairWalk
from src.mip.params import RepairWalkParams
from src.mip.propagation import LinearConstraintPropagator

from src.rl.architecture import PolicyArchitecture
from src.rl.params import GnnParams
from src.rl.learn import EvolutionaryStrategiesSerial, GradientAscent, FirstOrderTrainer, EsParallelTrajectories, \
    EsParallelInstances, Adam

from src.utils import initialize_logger

from src.rl.mip import EnhancedModel

import logging
import random

import gurobipy as gp
from gurobipy import GRB

from src.utils.config import *


def main():
    initialize_logger()

    # initialize architecture
    env = gp.Env()
    env.setParam(GRB.Param.OutputFlag, 0)
    m = gp.Model(env=env)
    logger: logging.Logger = logging.getLogger(__name__)
    logger.info('architecture created')

    x: gp.Var = m.addVar(ub=10, name='x', vtype=GRB.INTEGER)
    y: gp.Var = m.addVar(name='y', vtype=GRB.BINARY)
    z: gp.Var = m.addVar(ub=4, name='z', vtype=GRB.INTEGER)
    m.setObjective(2 * x + y, sense=GRB.MAXIMIZE)

    c1: gp.Constr = m.addConstr(x - y <= 1.75)
    c2: gp.Constr = m.addConstr(x + z == 1)

    m.presolve()
    m.optimize()

    fpr = FixPropRepair(None)

    for v in m.getVars():
        print(v)


def main2():
    random.seed(0)

    # this main file tests out FPRL
    size = 'medium'
    instance = 'random3sat_instance_0'
    mip_filename = f'/home/stevenmaio/PycharmProjects/rl-repair/data/instances/random3sat/{size}/{instance}.mps'
    input_policy = '/home/stevenmaio/PycharmProjects/rl-repair/data/torch_models/policy_architecture.pt'
    output_policy = '/home/stevenmaio/PycharmProjects/rl-repair/data/torch_models/policy_architecture-trained.pt'
    sample_indices: bool = False
    initialize_logger(level=logging.INFO)

    # initialize architecture
    env = gp.Env()
    env.setParam(GRB.Param.OutputFlag, 0)
    gp_model = gp.read(mip_filename, env)

    # create and load policy architecture
    policy_architecture = PolicyArchitecture(GnnParams)
    policy_architecture.load_state_dict(torch.load(input_policy))

    model = EnhancedModel.from_gurobi_model(gp_model,
                                            gnn=policy_architecture.gnn)
    repair_strat = LearnableRepairWalk(RepairWalkParams(),
                                       policy_architecture.cons_scoring_function,
                                       policy_architecture.var_scoring_function,
                                       sample_indices=sample_indices)
    fprl = FixPropRepairLearn(policy_architecture.fixing_order_architecture,
                              policy_architecture.value_fixing_architecture,
                              repair_strat,
                              LinearConstraintPropagator(),
                              policy_architecture,
                              sample_indices=sample_indices)
    success = fprl.find_solution(model)
    if success:
        for var in model.variables:
            print(f'x{var.id}={var.lb}')
        torch.save(policy_architecture.state_dict(), output_policy)
        print(f'reward={fprl.reward}')


def serial_es_main():
    initialize_logger(level=logging.INFO)

    instance_dir = os.sep.join([PROJECT_ROOT, INSTANCES])
    instances = [os.sep.join([instance_dir, f]) for f in os.listdir(instance_dir) if '.opb' in f or '.mps' in f][5:6]
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
                              discount_factor=DISCOUNT_FACTOR)

    gradient_estimator = EvolutionaryStrategiesSerial(num_trajectories=NUM_TRAJECTORIES,
                                                      learning_parameter=LEARNING_PARAMETER,
                                                      batch_size=BATCH_SIZE,
                                                      log_file=TRAINING_LOG)
    # optimization_method = GradientAscent(learning_rate=LEARNING_RATE)
    optimization_method = Adam(fprl=fprl,
                               step_size=LEARNING_RATE,
                               first_moment_decay_rate=FIRST_MOMENT_DECAY_RATE,
                               second_moment_decay_rate=SECOND_MOMENT_DECAY_RATE,
                               epsilon=EPSILON)
    trainer = FirstOrderTrainer(optimization_method=optimization_method,
                                num_epochs=NUM_EPOCHS,
                                gradient_estimator=gradient_estimator)
    trainer.train(fprl=fprl,
                  training_instances=instances,
                  save_rate=SAVE_RATE,
                  model_output=OUTPUT_MODEL)


def parallel_trajectories_es_main():
    initialize_logger(level=logging.INFO)

    instance_dir = os.sep.join([PROJECT_ROOT, INSTANCES])
    instances = [os.sep.join([instance_dir, f]) for f in os.listdir(instance_dir) if '.opb' in f or '.mps' in f]
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
                              discount_factor=DISCOUNT_FACTOR)

    gradient_estimator = EsParallelTrajectories(num_trajectories=NUM_TRAJECTORIES,
                                                learning_parameter=LEARNING_PARAMETER,
                                                num_workers=NUM_WORKERS)
    optimization_method = GradientAscent(learning_rate=LEARNING_RATE)
    trainer = FirstOrderTrainer(optimization_method=optimization_method,
                                num_epochs=NUM_EPOCHS,
                                gradient_estimator=gradient_estimator)
    trainer.train(fprl=fprl,
                  training_instances=instances,
                  save_rate=SAVE_RATE,
                  model_output=OUTPUT_MODEL)


def parallel_instances_es_main():
    initialize_logger(level=logging.INFO)

    instance_dir = os.sep.join([PROJECT_ROOT, INSTANCES])
    instances = [os.sep.join([instance_dir, f]) for f in os.listdir(instance_dir) if '.opb' in f or '.mps' in f]
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
                              discount_factor=DISCOUNT_FACTOR)

    gradient_estimator = EsParallelInstances(num_trajectories=NUM_TRAJECTORIES,
                                             learning_parameter=LEARNING_PARAMETER,
                                             num_workers=NUM_WORKERS)
    optimization_method = GradientAscent(learning_rate=LEARNING_RATE)
    trainer = FirstOrderTrainer(optimization_method=optimization_method,
                                num_epochs=NUM_EPOCHS,
                                gradient_estimator=gradient_estimator)
    trainer.train(fprl=fprl,
                  training_instances=instances,
                  save_rate=SAVE_RATE,
                  model_output=OUTPUT_MODEL)


if __name__ == '__main__':
    import time

    # start = time.time()
    # parallel_instances_es_main()
    # parallel_time = time.time() - start

    start = time.time()
    serial_es_main()
    serial_time = time.time() - start

    print(f'serial {serial_time}')
    # print(f'parallel {parallel_time}')
