import json
import argparse
import logging
import itertools

import torch

import gurobipy as gp
from gurobipy import GRB

from src.rl.mip import EnhancedModel
from src.rl.learn.trainer import trainer_from_config
from src.mip.heuristic import FixPropRepairLearn, heuristic_from_config
from src.rl.utils.DataSet import DataSet

from src.utils import initialize_logger, initialize_global_pool, get_global_pool, create_rng_seeds

from src.utils.config import *


def _eval_trajectory(fpr, instance, rng_seed):
    torch.set_num_threads(NUM_THREADS)
    torch.manual_seed(rng_seed)
    with torch.no_grad():
        if hasattr(fpr, "policy_architecture"):
            gnn = fpr.policy_architecture.gnn
        else:
            gnn = None
        env = gp.Env()
        env.setParam(GRB.Param.OutputFlag, 0)
        gp_model = gp.read(instance, env)
        model = EnhancedModel.from_gurobi_model(gp_model,
                                                gnn=gnn,
                                                convert_ge_cons=True)
        fpr.find_solution(model)
        return fpr.reward


def eval_main(args):
    logger = logging.getLogger()
    config_file = args.config_file
    with open(config_file) as f:
        config = json.load(f)

    initialize_logger(level=logging.INFO)

    # global config
    torch.set_num_threads(NUM_THREADS)
    initialize_global_pool(config[NUM_WORKERS])
    worker_pool = get_global_pool()
    rng_seed = config.get('rng_seed', None)
    if rng_seed is not None:
        torch.manual_seed(rng_seed)
    else:
        logger.info("rng_seed=%d", torch.initial_seed())

    data_set = DataSet.from_config(config[DATA_SET_CONFIG])
    heuristic = heuristic_from_config(config[HEURISTIC])

    num_trajectories = config["num_trajectories"]
    batch_size = len(data_set.testing_instances) * num_trajectories

    pool_inputs = [itertools.repeat(i, num_trajectories) for i in data_set.testing_instances]
    pool_inputs = zip(itertools.chain(*pool_inputs),
                      map(lambda t: t.item(), create_rng_seeds(batch_size)))

    results = worker_pool.starmap(_eval_trajectory,
                                  map(lambda t: (heuristic, t[0], t[1]),
                                      pool_inputs))
    num_successes = 0
    for r in results:
        if r > 0:
            num_successes += 1
    logger.info("success_rate=%.2f", num_successes / batch_size)


def train_main(args):
    config_file = args.config_file
    with open(config_file) as f:
        config = json.load(f)

    initialize_logger(level=logging.INFO)

    # global config
    model_output = config[MODEL_OUTPUT]
    torch.set_num_threads(NUM_THREADS)
    initialize_global_pool(config[NUM_WORKERS])
    rng_seed = config.get('rng_seed', None)
    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    data_set = DataSet.from_config(config[DATA_SET_CONFIG])
    fprl = FixPropRepairLearn.from_config(config[HEURISTIC])
    trainer = trainer_from_config(config[TRAINER_CONFIG])

    trainer.train(fprl=fprl,
                  data_set=data_set,
                  model_output=model_output)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='command',
                                       help='command for the solver',
                                       required=True)
    train_parser = subparsers.add_parser('train',
                                         help='train a model')
    train_parser.add_argument("config_file",
                              help="configuration file")
    train_parser.set_defaults(func=train_main)
    eval_parser = subparsers.add_parser('eval',
                                        help='evaluate a FPR[L] model')
    eval_parser.add_argument("config_file",
                             help="configuration file")
    eval_parser.set_defaults(func=eval_main)

    args = parser.parse_args()
    args.func(args)
