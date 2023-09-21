from typing import Tuple

import torch

from src.mip.heuristic.value.ValueFixingStrategy import ValueFixingStrategy


class RandomValueFixing(ValueFixingStrategy):
    name: str = "RandomValueFixing"

    def select_fixing_value(self, model: "Model", var: "Variable", generator=None) -> Tuple[int, int]:
        local_domain: "Domain" = var.local_domain
        lower_bound: int = int(local_domain.lower_bound)
        upper_bound: int = int(local_domain.upper_bound)
        if torch.rand(1, generator=generator) <= 0.5:
            left_value, right_value = lower_bound, upper_bound
        else:
            left_value, right_value = upper_bound, lower_bound
        return left_value, right_value

    def init(self, model, generator=None):
        ...
