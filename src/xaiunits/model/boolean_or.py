import itertools

import torch
import torch.nn as nn
from xaiunits.model.generic import generate_layers
from typing import Tuple


class BooleanOrNN(nn.Sequential):
    """
    Implements a neural network model designed to mimic the 'OR' logical operation on input features.

    When only -1 or 1 are passed in as part of the input to this model, it is effectively performing the
    'OR' operation.

    This model is structured as a sequence of layers that progressively compute the 'OR' operation
    on the input features, scaling the dimensionality of the input at each step until the final
    output is obtained. The network employs a specific arrangement of weights and intermediate operations
    such that it is also equivalent to computing the maximum value among a collection of values given as
    as the input.

    Inherits from:
        torch.nn.Sequential: Parent class for implementing neural networks with modules defined in a
        sequential manner.

    Attributes:
        dims (list): A list that keeps track of the dimensions of each layer in the network.
    """

    def __init__(self, n_features: int = 2) -> None:
        """
        Initializes an AND model with the specified input features dimension.

        Args:
            n_features (int): The dimension (number of features) of the input data. Defaults to 2.
        """
        assert n_features >= 2
        self.n_features = n_features
        layer_weights, layer_bias, layer_act_fns = self._create_layer_weights()
        layers = generate_layers(layer_weights, layer_bias, layer_act_fns)
        super().__init__(*layers)

    def _create_layer_weights(self) -> Tuple:
        """
        Initializes the layers of the network starting from the input dimension.

        Args:
            in_dim (int): The dimension of the input to the current layer.

        Returns:
            list: A list of initialized layers that make up the network.
        """

        all_combo = torch.tensor(
            list(itertools.product([1.0, -1.0], repeat=self.n_features))
        )
        mask = (all_combo == -1).all(dim=1)

        out = torch.tensor([1.0] * all_combo.shape[0])
        out[mask.nonzero()] = -1.0
        out = torch.diag_embed(out)
        out_bias = torch.ones((all_combo.shape[0])) * -(self.n_features - 1)
        final_out = torch.ones((1, all_combo.shape[0]))

        reform = torch.ones((1, 1)) * 2.0
        reform_bias = torch.ones(1) * -1.0

        return (
            [all_combo, out, final_out, reform],
            [None, out_bias, None, reform_bias],
            [nn.ReLU, nn.ReLU, None, None],
        )


if __name__ == "__main__":
    feature = 3
    all_combo = torch.tensor(list(itertools.product([1.0, -1.0], repeat=feature)))
    mask = (all_combo == -1).all(dim=1)
    model = BooleanOrNN(feature)
    res = model(all_combo)  # should return min
    print(mask.nonzero())
    print((res.squeeze() == -1).nonzero())
