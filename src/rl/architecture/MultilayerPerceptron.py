from torch import nn


class MultilayerPerceptron(nn.Module):
    _input_size: int
    _output_size: int
    _hidden_layer_size: int

    def __init__(self,
                 layers,
                 activation_function=nn.ReLU):
        """
        Creates a Multilayer Perceptron using ReLU activation functions. The sizes
        of the layers are determined by the layers list with the specification that
        layers[0] is the size of the input, and layers[-1] is the desired output
        sizes of the MLP.
        :param layers:
        """
        super().__init__()
        network_layers = []
        for input_size, output_size in zip(layers[:-2], layers[1:-1]):
            network_layers.append(nn.Linear(input_size, output_size))
            network_layers.append(activation_function())
        final_input_size, final_output_size = layers[-2:]
        network_layers.append(nn.Linear(final_input_size, final_output_size))
        self._network = nn.Sequential(*network_layers)

    def forward(self, x):
        return self._network(x)
