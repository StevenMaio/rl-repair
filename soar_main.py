import json
import argparse
import logging

import torch
import torch.multiprocessing as mp

from src.soar import SOAR
from src.soar.sampling import LatinHypercube
from src.soar.surrogate import SimpleGpSurrogate
from src.soar.restart import SampleMaxRestart
from src.soar.termination import AtMostKIters

from src.mip.heuristic import FixPropRepairLearn

from src.rl.utils import TensorList
from src.rl.learn import FirstOrderTrainer
from src.rl.utils.DataSet import DataSet

from src.utils import initialize_logger, initialize_global_pool

from src.utils.config import DATA_SET_CONFIG, NUM_THREADS, FPRL_CONFIG, NUM_WORKERS, TRAINER_CONFIG, MODEL_OUTPUT


def train_model(config_file):
    mp.set_start_method('forkserver')

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
    fprl = FixPropRepairLearn.from_config(config[FPRL_CONFIG])
    trainer = FirstOrderTrainer.from_config(config[TRAINER_CONFIG])

    local_search = trainer._optimization_method
    gradient_estimator = trainer._gradient_estimator

    parameters = TensorList(fprl.policy_architecture.parameters())
    corr_parameters = torch.ones(parameters.size) * 1_000.

    support = torch.zeros((parameters.size, 2))
    support[:, 0] = -1.0
    support[:, 1] = 1.0

    surrogate_model = SimpleGpSurrogate(corr_parameters, support, 10)
    experimental_design = LatinHypercube(support)
    termination_mechanism = AtMostKIters(10)

    restart_mechanism = SampleMaxRestart(best_ei_threshold=0.75,
                                         num_restart_samples=1_000,
                                         noise_paramter=0.0)

    soar = SOAR(local_search,
                gradient_estimator,
                surrogate_model,
                restart_mechanism,
                termination_mechanism,
                20_000,
                experimental_design,
                10,
                0,
                10,
                'soar-test-log.txt',
                True)
    soar.optimize(fprl, data_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file",
                        help="config file for training")
    args = vars(parser.parse_args())
    train_model(**args)
