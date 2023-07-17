"""
A class that separates a given collection of instances into three categories:
training, validation and testing.
"""
from typing import List

import random


class DataSet:
    _training_instances: List[object]
    _validation_instances: List[object]
    _testing_instances: List[object]

    def __init__(self, instances, validation_portion=0.2, testing_portion=0.2, rng_seed=None):
        rng = random.Random(rng_seed)
        N = len(instances)
        indices = list(range(N))

        val_size = int(validation_portion * N)
        test_size = int(testing_portion * N)

        removed_indices = random.sample(indices, val_size + test_size)
        val_indices = removed_indices[:val_size]
        test_indices = removed_indices[val_size:]

        for idx in removed_indices:
            indices.remove(idx)

        self._training_instances = [instances[idx] for idx in indices]
        self._validation_instances = [instances[idx] for idx in val_indices]
        self._testing_instances = [instances[idx] for idx in test_indices]

    @property
    def training_instances(self):
        return self._training_instances

    @property
    def validation_instances(self):
        return self._validation_instances

    @property
    def testing_indices(self):
        return self._testing_instances
