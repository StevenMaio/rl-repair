"""
A class that separates a given collection of instances into three categories:
training, validation and testing.
"""
import os
import json
import torch

from typing import List

from src.utils.config import PARAMS


class DataSet:
    _training_instances: List[object]
    _validation_instances: List[object]
    _testing_instances: List[object]

    def __init__(self, instances, validation_portion=0.2, testing_portion=0.2, rng_seed=None):
        rng = torch.Generator()
        if rng_seed is not None:
            rng.manual_seed(rng_seed)
        else:
            rng.seed()
        N = len(instances)

        val_size = int(validation_portion * N)
        test_size = int(testing_portion * N)

        permutation = torch.randperm(N, generator=rng)

        val_indices = permutation[:val_size]
        test_indices = permutation[val_size:val_size + test_size]
        training_indices = permutation[val_size + test_size:]

        self._training_instances = [instances[idx] for idx in training_indices]
        self._validation_instances = [instances[idx] for idx in val_indices]
        self._testing_instances = [instances[idx] for idx in test_indices]

    @property
    def training_instances(self):
        return self._training_instances

    @property
    def validation_instances(self):
        return self._validation_instances

    @property
    def testing_instances(self):
        return self._testing_instances

    @staticmethod
    def from_config(config: dict):
        params = config[PARAMS]
        instances_location = config['instances']
        instances = [os.sep.join([instances_location, f]) for f in os.listdir(instances_location) if '.opb' in f or '.mps' in f]
        return DataSet(instances, **params)
