import torch
import torch.nn as nn
from xaiunits.model.generic import generate_layers
from typing import List, Tuple, Union, Dict


class InteractingFeaturesNN(nn.Sequential):
    """
    Implements a neural network model designed to explicitly model interactions between specific pairs of features
    within the input data.

    This model is capable of emphasizing or de-emphasizing the impact of these interactions
    on the model's output through a specialized network architecture and custom weight assignments.

    The network consists of linear layers combined with ReLU activation, structured to manipulate the input
    # features based on the predefined interactions. The interactions are modelled such that the influence of one
    feature on another can be either enhanced or canceled, according to the specified weights and the interaction
    mechanism implemented within the network.

    Inherits from:
        torch.nn.Sequential: Parent class for implementing neural networks with modules defined in a
        sequential manner.
    """

    def __init__(
        self,
        n_features: int,
        weights: List[Union[float, Tuple]],
        interacting_features: List[Tuple[int, int]],
    ) -> None:
        """
        Initializes the InteractingFeaturesNN model with specified dimensions, weights, and interactions.

        The architecture is designed to create a network that can process feature interactions by rearranging
        and weighting input features according to the specified interactions.

        Args:
            n_features (int): The total number of features in the input data. This includes both interacting
                and non-interacting features.
            weights (list) : A list of floats or tuples specifying the weights to be applied to the features of
                the model. This list should have a len that matches the `n_features`, with each element
                corresponding to a feature in the input data.
            interacting_features (list[tuple]): A list where each tuple contains two integers representing
                the indices of the interacting features. The first element in the tuple is considered the impacting
                feature, and the second element is the feature being impacted.
        """
        self._validate_inputs(weights, interacting_features)
        layer_weights, layer_bias, layer_act_fns = self._create_layer_weights(
            n_features, weights, interacting_features
        )
        layers = generate_layers(layer_weights, layer_bias, layer_act_fns)

        super().__init__(*layers)

    def _validate_inputs(
        self,
        weights: List[Union[float, Tuple]],
        interacting_features: List[Tuple[int, int]],
    ) -> None:
        """
        Validates the inputs.

        Args:
            weights (list) : A list of floats or tuples specifying the weights to be applied to the features of
                the model.
            interacting_features (list[tuple]): A list where each tuple contains two integers representing
                the indices of the interacting features.

        Raises:
            AssertionError: If the inputs are not in the valid datatypes.
        """
        for impacts, impacted in interacting_features:
            assert type(weights[impacts]) in [float, int]
            assert type(weights[impacted]) == tuple

    def _create_layer_weights(
        self,
        n_features: int,
        weights: List[Union[float, Tuple]],
        interacting_features: List[Tuple[int, int]],
    ) -> Tuple:
        """
        Creates the weights for the layers in a InteractingFeaturesNN model.

        Args:
            n_features (int): The total number of features in the input data.
            weights (list) : A list of floats or tuples specifying the weights to be applied to the features of
                the model.
            interacting_features (list[tuple]): A list where each tuple contains two integers representing
                the indices of the interacting features.

        Returns:
            tuple[list, NoneType, list]: Tuple containing the weights and activation functions for the
            neural network model.
        """
        N = n_features
        large_factor = -1000.0

        all_impacts = [i[0] for i in interacting_features]
        all_impacted = [i[1] for i in interacting_features]
        others = [i for i in range(N) if not (i in all_impacted or i in all_impacts)]

        # Initialize w0 and layer 0
        in_ordering = list(range(N))
        out_ordering = all_impacts * 2 + all_impacted * 4 + others * 2
        out_ordering.sort()
        w_0 = torch.zeros((len(out_ordering), len(in_ordering)))
        filled = set()
        for impacts, impacted in interacting_features:
            in_impacts_indices = [i for i, x in enumerate(in_ordering) if x == impacts]
            in_impacted_indices = [
                i for i, x in enumerate(in_ordering) if x == impacted
            ]
            out_impacts_indices = [
                i for i, x in enumerate(out_ordering) if x == impacts
            ]
            out_impacted_indices = [
                i for i, x in enumerate(out_ordering) if x == impacted
            ]

            inter = torch.Tensor(
                [weights[impacted][0] - weights[impacted][1], weights[impacted][1]]
            )
            inter = torch.cat([inter, -1 * inter], dim=0)
            w_0[out_impacted_indices, in_impacted_indices] = inter
            w_0[out_impacts_indices, in_impacts_indices] = torch.Tensor(
                [weights[impacts], -1 * weights[impacts]]
            )
            w_0[
                [out_impacted_indices[0], out_impacted_indices[2]], in_impacts_indices
            ] = torch.Tensor([large_factor, large_factor])
            filled = filled | {impacts, impacted}

        unfilled = [i for i in range(N) if i not in filled]
        for unfill in unfilled:
            in_unfill_indices = [i for i, x in enumerate(in_ordering) if x == unfill]
            out_unfill_indices = [i for i, x in enumerate(out_ordering) if x == unfill]
            w_0[out_unfill_indices, in_unfill_indices] = torch.Tensor(
                [weights[unfill], -weights[unfill]]
            )

        w_0 = w_0.float()
        a0 = nn.ReLU

        w_1 = torch.zeros((1, len(out_ordering)))
        filled = set()
        for impacts, impacted in interacting_features:
            out_impacts_indices = [
                i for i, x in enumerate(out_ordering) if x == impacts
            ]
            out_impacted_indices = [
                i for i, x in enumerate(out_ordering) if x == impacted
            ]

            inter = torch.tensor([1.0, 1.0, -1.0, -1.0])
            w_1[0, out_impacted_indices] = inter
            w_1[0, out_impacts_indices] = torch.Tensor([1.0, -1.0])
            filled = filled | {impacts, impacted}

        unfilled = [i for i in range(N) if i not in filled]
        for unfill in unfilled:
            out_unfill_indices = [i for i, x in enumerate(out_ordering) if x == unfill]
            w_1[0, out_unfill_indices] = torch.Tensor([1.0, -1.0])

        return [w_0, w_1], None, [a0, None]


if __name__ == "__main__":
    from captum.attr import InputXGradient
    from xaiunits.datagenerator import (
        InteractingFeatureDataset,
    )
    from xaiunits.methods import wrap_method

    data = InteractingFeatureDataset(n_samples=1000)
    model = data.generate_model()

    print(data.weights)
    print(sum(data.weights[0]))
    print(model(torch.tensor([10.0, 0.0, 0.0, 0.0])))
