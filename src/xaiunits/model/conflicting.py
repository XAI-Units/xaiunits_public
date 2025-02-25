import torch
import torch.nn as nn
from xaiunits.model.generic import generate_layers
from typing import Tuple


class ConflictingFeaturesNN(nn.Sequential):
    """
    A crafted neural network model that incorporates cancellation features.

    Inherits from:
        torch.nn.Sequential: Parent class for implementing neural networks with modules defined in a
        sequential manner.
    """

    def __init__(self, continuous_dim: int, weights: torch.Tensor) -> None:
        """
        Initializes a ConflictingFeaturesNN object.

        Args:
            continuous_dim (int): Dimension length of the continuous features, excluding cancellation features.
            weights (torch.Tensor): Feature weights of the model. Should have length `continuous_dim`.
        """
        layer_weights, layer_bias, layer_act_fns = self._create_layer_weights(
            continuous_dim, weights
        )
        layers = generate_layers(layer_weights, layer_bias, layer_act_fns)

        super().__init__(*layers)

    def _create_layer_weights(
        self, continuous_dim: int, weights: torch.Tensor
    ) -> Tuple:
        """
        Creates the weights for the layers in a ConflictingFeaturesNN model.

        Args:
            continuous_dim (int): Number of continuous features.
            weights (torch.Tensor): Feature weights of the model. Should have length `continuous_dim`.

        Returns:
            tuple[list, NoneType, list]: Tuple containing the weights and activation functions for the
            neural network model.

        Raises:
            AssertionError: If weights are not specified in a valid shape.
        """
        large_weight = 100.0

        if weights.dim() == 1:
            assert len(weights) == continuous_dim
        elif weights.dim() == 2:
            assert weights.shape[1] == continuous_dim

        # Weights0 -> split x into pos x and neg -x, multiply by input weight, add -c*large_weight to both
        w0 = torch.zeros((2 * continuous_dim, 2 * continuous_dim))
        d = continuous_dim
        w0[:d, :d] = torch.diag_embed(weights)
        w0[:d, d:] = -large_weight * torch.eye(d)
        w0[d:, :d] = -torch.diag_embed(weights)
        w0[d:, d:] = -large_weight * torch.eye(d)

        # Weights1 -> [sum of un-cancelled xs]
        w1 = torch.zeros((1, continuous_dim * 2))
        w1[0, :continuous_dim] = 1.0
        w1[0, continuous_dim : 2 * continuous_dim] = -1.0

        return [w0, w1], None, [nn.ReLU, None]


if __name__ == "__main__":
    # Create model
    continuous_dim = 2
    weights = torch.tensor([3.2, 3.5])
    model = ConflictingFeaturesNN(continuous_dim, weights)

    # Create input
    x = torch.tensor([-1.0, 2.0, 1.0, 0.0])

    # Get output
    y = model(x)
    print(y)
    # for params in model.parameters():
    #     print(params)
