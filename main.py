import json
import argparse
import logging

import torch
import torch.multiprocessing as mp

from src.rl.learn import FirstOrderTrainer
from src.mip.heuristic import FixPropRepairLearn
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

    trainer.train(fprl=fprl,
                  data_set=data_set,
                  model_output=model_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file",
                        help="config file for training")
    args = vars(parser.parse_args())
    train_model(**args)
