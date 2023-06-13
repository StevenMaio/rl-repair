from src.mip.heuristic import FixPropRepairLearn
from src.mip.heuristic.repair import LearnableRepairWalk
from src.mip.params import RepairWalkParams
from src.mip.propagation import LinearConstraintPropagator
from src.rl.architecture import GraphNeuralNetwork, MultilayerPerceptron
from src.rl.params import GnnParams
from src.mip.heuristic import FixPropRepair
from src.utils import initialize_logger

from src.rl.mip import EnhancedModel

import logging

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
    # this main file tests out FPRL
    mip_filename = '/home/stevenmaio/PycharmProjects/rl-repair/instances/random3sat/medium/random3sat_instance_0.mps'
    initialize_logger()

    # initialize architecture
    env = gp.Env()
    env.setParam(GRB.Param.OutputFlag, 0)
    gp_model = gp.read(mip_filename, env)
    gp_model.presolve()
    model = EnhancedModel.from_gurobi_model(gp_model)

    gnn = GraphNeuralNetwork(GnnParams)
    model.gnn = gnn
    num_learned_node_features = GnnParams.intermediate_layers
    num_edge_features = GnnParams.num_edge_features

    cons_scoring_function = MultilayerPerceptron([num_learned_node_features,
                                                  64,
                                                  1])
    var_scoring_function = MultilayerPerceptron([2 * num_learned_node_features + num_edge_features,
                                                 64,
                                                 1])

    repair_strat = LearnableRepairWalk(RepairWalkParams(),
                                       cons_scoring_function,
                                       var_scoring_function)
    fprl = FixPropRepairLearn(repair_strat,
                              LinearConstraintPropagator())
    success = fprl.find_solution(model)
    if success:
        for var in model.variables:
            print(f'x{var.id}={var.lb}')


if __name__ == '__main__':
    #main()
    # parser = init_parser()
    # parser.parse_args()
    main2()
