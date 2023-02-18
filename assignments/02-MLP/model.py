import torch
from typing import Callable


class MLP(torch.nn.Module):
    """
    A multilayer perceptron model
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            hidden_count: The number of hidden layers.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.activation = activation()
        self.hidden_layers = [torch.nn.Linear(input_size, hidden_size)]
        initializer(self.hidden_layers[0].weight)
        for _ in range(1, hidden_count):
            self.hidden_layers.append(torch.nn.Linear(hidden_size, hidden_size))
            initializer(self.hidden_layers[-1].weight)
        self.output_layer = torch.nn.Linear(hidden_size, num_classes)
        initializer(self.output_layer.weight)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        return self.output_layer(x)
