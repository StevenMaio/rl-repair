"""
An implementation of Adam for maximization problems.

Sources:
[1] D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization.”
    arXiv, Jan. 29, 2017. Accessed: Jul. 11, 2023. [Online].
    Available: http://arxiv.org/abs/1412.6980
"""
import torch

from .FirstOrderMethod import FirstOrderMethod
from src.rl.utils import TensorList


class Adam(FirstOrderMethod):
    _step_size: float
    _fmdr: float        # first moment decay rate
    _1st_mom_decay: torch.Tensor
    _smdr: float        # second moment decay rate
    _2nd_mom_decay: torch.Tensor
    _iter_num: float
    _first_moment_estimate: TensorList
    _second_moment_estimate: TensorList
    _epsilon: torch.Tensor

    def __init__(self,
                 fprl: "FPRL",
                 step_size: float,
                 first_moment_decay_rate: float,
                 second_moment_decay_rate: float,
                 epsilon: float = 1e-9):
        policy_architecture = fprl.policy_architecture
        self._step_size = step_size
        self._fmdr = first_moment_decay_rate
        self._1st_mom_decay = torch.Tensor([1.0])
        self._smdr = second_moment_decay_rate
        self._2nd_mom_decay = torch.Tensor([1.0])
        self._iter_num = 0
        self._epsilon = torch.Tensor([epsilon])
        self._first_moment_estimate = TensorList.zeros_like(policy_architecture.parameters())
        self._second_moment_estimate = TensorList.zeros_like(policy_architecture.parameters())

    def reset(self):
        self._iter_num = 0
        self._first_moment_estimate.zero_out()
        self._second_moment_estimate.zero_out()

    def step(self, policy_architecture, gradient_estimate):
        """
        See [1] for update rule and description.

        :param policy_architecture:
        :param gradient_estimate:
        :return:
        """
        with torch.no_grad():
            self._iter_num += 1
            self._1st_mom_decay *= self._fmdr
            self._2nd_mom_decay *= self._smdr
            for g, m, v, p in zip(gradient_estimate,
                                  self._first_moment_estimate,
                                  self._second_moment_estimate,
                                  policy_architecture.parameters()):
                # update 1st moment estimate
                m.mul_(self._fmdr)
                m.add_((1 - self._fmdr) * g)
                # update 2nd moment estimate
                v.mul_(self._smdr)
                v.add_((1 - self._smdr) * g.square())
                # compute corrected 1st moment estimate
                m_hat = m / (1 - self._1st_mom_decay)
                # compute corrected 2nd moment estimate
                v_hat = v / (1 - self._2nd_mom_decay)
                p.add_(self._step_size * m_hat / (v_hat.sqrt_() + self._epsilon))
