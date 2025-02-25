import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple


def _validate_and_reformat_inputs(
    weights: Union[torch.Tensor, List[torch.Tensor]],
    biases: Union[torch.Tensor, List[torch.Tensor]],
    act_fns: Union[torch.nn.Module, List[torch.nn.Module]],
) -> Tuple:
    """
    Validates and reformats inputs.

    Args:
        weights (torch.Tensor | list[torch.Tensor]): Input weight tensors.
        biases (torch.Tensor | list[torch.Tensor] | NoneType): Input bias tensors.
        act_fns (torch.Tensor | list[torch.Tensor] | NoneType): Activation functions.

    Returns:
        tuple: Tuple contianing the reformatted weights, biases, and activation functions.

    Raises:
        TypeError: If inputs are not of the expected types.
        AssertionError: If dimensions of input tensors are incorrect or mismatched.
    """
    if type(weights) == torch.Tensor:
        assert weights.dim() < 3
        if weights.dim() == 1:
            weights = weights.reshape(1, -1)
        weights = [weights]
    elif type(weights) not in [list]:
        raise TypeError("Weights should be either List or torch.Tensor")
    else:
        for i, w in enumerate(weights):
            assert type(w) == torch.Tensor
            assert w.dim() < 3
            if w.dim() == 1:
                weights[i] = w.reshape(1, -1)

    if biases is not None:
        if type(biases) == torch.Tensor:
            assert biases.dim() == 1
            biases = [biases]
        elif type(biases) not in [list]:
            raise TypeError("Biases should be either List or torch.Tensor")
        else:
            for b in biases:
                if b is not None:
                    assert type(b) == torch.Tensor
                    assert b.dim() == 1

        assert len(weights) == len(biases)
    else:
        biases = [None] * len(weights)

    if act_fns is not None:
        if getattr(act_fns, "__module__", None) == "torch.nn.modules.activation":
            act_fns = [act_fns] * len(weights)
        elif type(act_fns) not in [list]:
            raise TypeError(
                "Activation Functions should be either List or part of nn.modules.activation"
            )
        else:
            for fn in act_fns:
                if fn is not None:
                    assert (
                        getattr(fn, "__module__", None) == "torch.nn.modules.activation"
                    )

        assert len(act_fns) == len(weights)
    else:
        act_fns = [None] * len(weights)

    return weights, biases, act_fns


def generate_layers(
    weights: Union[torch.Tensor, List[torch.Tensor]],
    biases: Union[torch.Tensor, List[torch.Tensor]],
    act_fns: Union[torch.nn.Module, List[torch.nn.Module]],
) -> List[nn.Module]:
    """
    Creates linear layers and activation function to be used as inputs for nn.Sequential.

    Args:
        weights (torch.Tensor | list[torch.Tensor]): Weights for each linear layer.
        biases (torch.Tensor | list[torch.Tensor] | NoneType): Bias for each linear layer.
            Length of weights and biases must match if list.
        act_fns (nn.module.activation | list | NoneType): Activation for each linear layer.
            If activation Layer (i.e. not list) act_fns will be repeated for each linear layer.
            It is recommended to pass in the class rather than instance of activation Layer,
            as certain FA methods require no duplicate layers in the model.
    """
    weights, biases, act_fns = _validate_and_reformat_inputs(weights, biases, act_fns)
    layers = []
    for weight, bias, fn in zip(weights, biases, act_fns):
        lin_layer = nn.Linear(
            in_features=weight.shape[1],
            out_features=weight.shape[0],
            bias=bias is not None,
        )
        lin_layer.weight = nn.Parameter(weight).float()

        if bias is not None:
            lin_layer.bias = nn.Parameter(bias).float()

        layers.append(lin_layer)

        if fn is not None:
            if type(fn) == type:
                layers.append(fn())
            else:
                layers.append(fn)

    return layers


class GenericNN(nn.Sequential):
    """
    Class for creating custom neural network architectures using specified weights,
    biases, and activation functions.

    Inherits from:
        torch.nn.Sequential: Parent class for implementing neural networks with modules defined in a
        sequential manner.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        biases: Optional[torch.Tensor] = None,
        act_fns: Optional[torch.nn.Module] = None,
    ) -> None:
        """
        Initializes a GenericNN object.

        Args:
            weights (torch.Tensor | list[torch.Tensor]): Weights for each linear layer.
            biases (torch.Tensor | list[torch.Tensor], optional): Bias for each linear layer.
                Length of weights and biases must match if list.
            act_fns (nn.module.activation | list, optional): Activation for each linear layer.
        """
        layers = generate_layers(weights, biases, act_fns)
        super().__init__(*layers)


if __name__ == "__main__":

    w = torch.Tensor([[1, 1, 1, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    print(type(getattr(nn.ReLU, "__module__", None)))
    model = GenericNN(w, act_fns=nn.Softmax)
    output = model(torch.tensor([[0, 1, 0, 0]]).float())
    print(output)
