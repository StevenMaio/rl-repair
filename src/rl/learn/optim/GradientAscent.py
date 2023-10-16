import torch

from .FirstOrderMethod import FirstOrderMethod


class GradientAscent(FirstOrderMethod):
    _learning_rate: float

    def __init__(self, learning_rate: float):
        self._learning_rate = learning_rate

    def step(self, policy_architecture, gradient_estimate):
        gradient_estimate.scale(self._learning_rate)
        with torch.no_grad():
            gradient_estimate.add_to_iterator(policy_architecture.parameters())
