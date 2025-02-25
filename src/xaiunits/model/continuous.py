import torch
import torch.nn as nn
from xaiunits.model.generic import generate_layers
from typing import Tuple


class ContinuousFeaturesNN(nn.Sequential):
    """
    A crafted neural network model that incorporates continuous features with ReLU.

    Inherits from:
        torch.nn.Sequential: Parent class for implementing neural networks with modules defined in a
        sequential manner.
    """

    def __init__(self, n_features: int, weights: torch.Tensor) -> None:
        """
        Initializes a ContinuousFeaturesNN object.

        Args:
            n_features (int): dimensions of the input.
            weights (torch.Tensor): weights of the model. Should have length `n_features`.
        """
        layer_weights, layer_bias, layer_act_fns = self._create_layer_weights(
            n_features, weights
        )
        layers = generate_layers(layer_weights, layer_bias, layer_act_fns)

        super().__init__(*layers)

    def _create_layer_weights(self, n_features: int, weights: torch.Tensor) -> Tuple:
        """
        Creates the weights for the layers in a ContinuousFeaturesNN model.

        Args:
            n_features (int): Number of features.
            weights (torch.Tensor): Feature weights of the model. Should have length `continuous_dim`.

        Returns:
            tuple[list, NoneType, list]: Tuple containing the weights and activation functions for the
            neural network model.

        Raises:
            AssertionError: If weights are not specified in a valid shape.
        """
        if weights.dim() == 1:
            assert len(weights) == n_features
        elif weights.dim() == 2:
            assert weights.shape[1] == n_features

        # weights for l0
        w0 = torch.zeros((2 * n_features, n_features))
        w0[:n_features] = torch.diag_embed(weights)
        w0[n_features : 2 * n_features] = torch.diag_embed(-weights)

        # weights for l2
        w1 = torch.ones((1, 2 * n_features))
        w1[:, n_features:] = -1.0

        return [w0, w1], None, [nn.ReLU, None]
