import torch
import os

from src.mip.heuristic import FixPropRepairLearn
from src.mip.heuristic.repair import LearnableRepairWalk
from src.mip.params import RepairWalkParams
from src.mip.propagation import LinearConstraintPropagator
from src.mip.heuristic import FixPropRepair

from src.rl.architecture import PolicyArchitecture
from src.rl.params import GnnParams
from src.rl.learn import EvolutionaryStrategies

from src.utils import initialize_logger

from src.rl.mip import EnhancedModel

import logging
import random

import gurobipy as gp
from gurobipy import GRB


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

def main3():
    # get training instances
    instance_dir = '/home/stevenmaio/PycharmProjects/rl-repair/data/instances/random3sat/small'
    instances = [os.sep.join([instance_dir, f]) for f in os.listdir(instance_dir)]
    input_policy = '/home/stevenmaio/PycharmProjects/rl-repair/data/torch_models/k-clique.pt'
    policy_output = '/home/stevenmaio/PycharmProjects/rl-repair/data/torch_models/k-clique.pt'
    initialize_logger(level=logging.INFO)

    # create and load policy architecture
    sample_indices: bool = True
    policy_architecture = PolicyArchitecture(GnnParams)
    policy_architecture.load_state_dict(torch.load(input_policy))
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

    # configure training algorithm
    num_epochs = 25
    num_trajectories = 10
    learning_parameter = 5
    learning_algorithm = EvolutionaryStrategies(num_epochs,
                                                num_trajectories,
                                                learning_parameter)
    with torch.no_grad():
        learning_algorithm.train(fprl, instances)
        print(learning_algorithm._num_successes)
    torch.save(policy_architecture.state_dict(), policy_output)


if __name__ == '__main__':
    #main()
    main3()
