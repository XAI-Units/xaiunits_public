import torch.nn as nn
import torch
from xaiunits.model import generate_layers


class ShatteredGradientsNN(nn.Sequential):
    """
    Implements a neural network model using a linear layer followed by an activation function.

    This model is designed to exhibit shattered gradients. To generate a model that exhibits
    shattered gradients, use shattered_grad.py, located in datagenerator.

    Inherits from:
        torch.nn.Sequential: Parent class for implementing neural networks with modules defined in a
        sequential manner.
    """

    def __init__(self, weights: torch.Tensor, act_fun: str = "Relu") -> None:
        """
        Initializes a ShatteredGradientsNN object.

        Args:
            weights (torch.Tensor): The weights to be applied to the linear layer.
            act_fun (str): The activation function to be used. Valid options are "Relu", "Gelu", or "Sigmoid".
                Defaults to 'Relu'.
        """
        act_fun = self._default_activation_function(act_fun)
        layers = generate_layers(weights=weights, biases=None, act_fns=act_fun)

        super().__init__(*layers)

    def _default_activation_function(self, act_fun: str) -> nn.Module:
        """
        Returns the default activation function based on the provided string.

        The `_default_activation_function` method maps the provided activation function name
        to the corresponding PyTorch activation function module. Supported activation functions
        include "Relu" (ReLU), "Gelu" (GELU), and "Sigmoid" (Sigmoid). If the provided name
        is not in the supported list, a KeyError is raised.

        Args:
            act_fun (str): The activation function name or class.

        Returns:
            torch.nn.Module: The corresponding PyTorch activation function module.

        Raises:
            KeyError: If the given activation function is not supported.
        """
        mapping = {"Relu": nn.ReLU, "Gelu": nn.GELU, "Sigmoid": nn.Sigmoid}

        if mapping.get(act_fun, False):
            act_fun = mapping[act_fun]
        elif act_fun in mapping.values():
            act_fun = act_fun
        else:
            raise KeyError(
                "Activation Function can only be of Type Relu, Gelu or Sigmoid."
            )

        return act_fun


if __name__ == "__main__":
    import torch
    from captum.attr import IntegratedGradients

    # Create model
    weights = torch.tensor([4000.05, 10000, 4000.05, 4000.05, 4000.05])
    act_fun = "Relu"
    model = ShatteredGradientsNN(weights, act_fun)

    # Create input
    x1 = torch.tensor([-1.001, 4, -2, -5, -2])
    x2 = torch.tensor([-0.999, 4, -2, -5, -2])

    # Get output
    y1 = model(x1)
    y2 = model(x2)

    # Get attributions
    ig = IntegratedGradients(model)
    attr_x1 = ig.attribute(x1.unsqueeze(0), target=0)
    attr_x2 = ig.attribute(x2.unsqueeze(0), target=0)

    # Results
    print("x1: ", x1)
    print("model forward output for x1: ", y1)
    print("Attributions for x1: ", attr_x1)

    print("x2: ", x2)
    print("model forward output for x2: ", y2)
    print("Attributions for x2: ", attr_x2)
