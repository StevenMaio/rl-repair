import torch
from torch import nn

from src.rl.params import GnnParams
from src.rl.architecture import MultilayerPerceptron
from src.rl.architecture.GraphNeuralNetworkV2 import GraphNeuralNetworkV2 as GraphNeuralNetwork


class PolicyArchitecture(nn.Module):
    _gnn: GraphNeuralNetwork
    _fixing_order_architecture: MultilayerPerceptron
    _value_fixing_architecture: MultilayerPerceptron
    _cons_scoring_function: MultilayerPerceptron
    _var_scoring_function: MultilayerPerceptron

    def __init__(self, params):
        """
        Initializes the policy architecture. Can load from a file, and save to a file as well.

        :param params:
        """
        super().__init__()
        num_learned_node_features = params.intermediate_layers
        num_edge_features = params.num_edge_features

        self._fixing_order_architecture = MultilayerPerceptron([num_learned_node_features,
                                                                2 * num_learned_node_features,
                                                                1])
        self._value_fixing_architecture = MultilayerPerceptron([num_learned_node_features,
                                                                2 * num_learned_node_features,
                                                                1])
        self._cons_scoring_function = MultilayerPerceptron([num_learned_node_features,
                                                            64,
                                                            1])
        self._var_scoring_function = MultilayerPerceptron([2 * num_learned_node_features + num_edge_features
                                                                + params.additional_var_scoring_features,
                                                           64,
                                                           1])
        self._gnn = GraphNeuralNetwork(params)

    def forward(self, x):
        raise NotImplementedError("forward not implemented for this class")

    @property
    def gnn(self):
        return self._gnn

    @property
    def fixing_order_architecture(self):
        return self._fixing_order_architecture

    @property
    def value_fixing_architecture(self):
        return self._value_fixing_architecture

    @property
    def cons_scoring_function(self):
        return self._cons_scoring_function

    @property
    def var_scoring_function(self):
        return self._var_scoring_function

    @staticmethod
    def from_config(config: dict):
        policy_architecture = PolicyArchitecture(GnnParams)
        if config['load_architecture'] and 'input_model' in config:
            policy_architecture.load_state_dict(torch.load(config['input_model']))
        return policy_architecture
