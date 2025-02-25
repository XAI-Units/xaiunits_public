import torch
import torch.nn as nn
from xaiunits.model.generic import generate_layers
from typing import Tuple


class PertinentNN(nn.Sequential):
    """
    Implements a neural network model specifically designed to handle pertinent negatives in input features.

    This model modifies input features based on their relevance and the presence of pertinent negatives,
    employing a specialized network architecture with custom weights and biases to emphasize or suppress
    certain features according to their pertinence.

    The network consists of linear layers combined with ReLU activation functions, structured to manipulate
    the input features dynamically. It uses the provided weights and a 'pertinence' tensor to adjust the
    impact of each feature on the model's output, effectively highlighting the role of
    pertinent negatives in the prediction process.


    Inherits from:
        torch.nn.Sequential: Parent class for implementing neural networks with modules defined in a
        sequential manner.
    """

    def __init__(
        self,
        n_features: int,
        weights: torch.Tensor,
        pn_features: torch.Tensor,
        pn_weight_factor: float,
    ) -> None:
        """
        Initializes the PertinentNN model with specified dimensions, weights, pertinent negatives, and a multiplier.

        The architecture is designed to first adjust the input features based on their pertinence, then to process
        these adjusted features through a series of layers that further manipulate and combine them based on the
        specified weights and the multiplier for pertinent negatives. The final output is a single value obtained
        through a linear combination of the processed features.

        Args:
            n_features (int): The total number of features in the input data.
            weights (torch.Tensor): A tensor specifying the weights to be applied to the features of the model.
                This tensor should have a shape that matches the `n_features`, with each weight corresponding to
                a feature in the input data.
            pn_features (torch.Tensor): A tensor indicating the presence (1) or absence (0) of pertinent negatives for each
                feature. The length of this tensor can be equal to or less than `n_features`. If it is less, the missing
                values are assumed to be 0 (no pertinent negative).
            pn_weight_factor (float): A multiplier used to adjust the weights of the features identified as pertinent negatives.
        """

        pn_features = self._reformat_pn_weight(n_features, pn_features)
        layer_weights, layer_bias, layer_act_fns = self._create_layer_weights(
            n_features, weights, pn_features, pn_weight_factor
        )
        layers = generate_layers(layer_weights, layer_bias, layer_act_fns)

        super().__init__(*layers)

    def _reformat_pn_weight(
        self, n_features: int, pn_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Reformats pn_features into tensors.

        Args:
            n_features (int): The total number of features in the input data.
            pn_features (torch.Tensor): A torch.Tensor object indicating which feature is a pertinent negative.

        Returns:
            torch.Tensor: Reformatted tensor representing the features.
        """
        if len(pn_features) < n_features:
            pn_features = torch.stack(
                (pn_features, torch.zeros(n_features - len(pn_features)))
            ).reshape(-1)
        if len(pn_features) > n_features:
            pn_features = pn_features[: len(pn_features) - n_features + 1]

        return pn_features

    def _create_layer_weights(
        self,
        n_features: int,
        weights: torch.Tensor,
        pn_features: torch.Tensor,
        pn_weight_factor: float,
    ) -> Tuple:
        """
        Creates the weights for the layers in a PertinentNN model.

        Args:
            n_features (int): The total number of features in the input data.
            weights (torch.Tensor): A tensor specifying the weights to be applied to the features of the model.
            pn_features (torch.Tensor): A tensor indicating the presence (1) or absence (0) of pertinent negatives for each
                feature.
            pn_weight_factor (float): A multiplier used to adjust the weights of the features identified as pertinent negatives.

        Returns:
            tuple[list, list, list]: Tuple containing the weights and activation functions for the
            neural network model.
        """
        layer_weights = []
        layer_bias = []
        layer_act_fns = []

        # First Layer,
        layer_weights.append(torch.eye(n_features) + torch.diag_embed(pn_features))
        layer_bias.append(pn_features * -1)
        layer_act_fns.append(None)

        # Second Layer
        w1 = torch.zeros((n_features * 2, n_features))
        w1[:n_features, :n_features] = torch.eye(n_features)
        w1[n_features : 2 * n_features, :n_features] = -torch.eye(n_features)
        layer_weights.append(w1)
        layer_bias.append(None)
        layer_act_fns.append(nn.ReLU)

        # Third Layer
        w2 = torch.zeros((2 * n_features, 2 * n_features))
        w2[:n_features, :n_features] = torch.diag_embed(weights)
        w2[n_features:, n_features:] = torch.diag_embed(
            (weights * -1 * (pn_features - 1))
            + weights * pn_features * pn_weight_factor
        )
        layer_weights.append(w2)
        layer_bias.append(None)
        layer_act_fns.append(None)

        # Forth Layer
        w3 = torch.zeros((1, n_features * 2))
        w3[:, :n_features] = torch.ones(n_features)
        w3[:, n_features:] = pn_features * 2 - 1
        layer_weights.append(w3)
        layer_bias.append(None)
        layer_act_fns.append(None)

        return layer_weights, layer_bias, layer_act_fns


if __name__ == "__main__":
    from captum.attr import DeepLift
    from xaiunits.datagenerator import (
        PertinentNegativesDataset,
    )

    data = PertinentNegativesDataset()
    print(data.pn_weight_factor)
    print(data.weights)

    model = data.generate_model()
    print(model(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])))

    print(data[:2])
    print(data)
    print(model(data[:2][0]))
    # self.n_features, self.weights, self.PN, self.pn_weight_factor
