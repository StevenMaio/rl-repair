from src.mip.heuristic import FixPropRepair
from src.utils import initialize_logger

import logging

import gurobipy as gp
from gurobipy import GRB


def main():
    initialize_logger()

    # initialize model
    env = gp.Env()
    env.setParam(GRB.Param.OutputFlag, 0)
    m = gp.Model(env=env)
    logger: logging.Logger = logging.getLogger(__name__)
    logger.info('model created')

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


if __name__ == '__main__':
    main()
