import torch
import torch.nn as nn
from typing import List, Dict, Optional


class DynamicNN(nn.Sequential):
    """
    Class that enables the instantiation of custom neural network architectures using a list
    of layer configurations.

    Inherits from:
        torch.nn.Sequential: Parent class for implementing neural networks with modules defined in a
        sequential manner.
    """

    def __init__(self, config: List[Dict], custom_layers: Optional[List] = None):
        """
        Initializes a DynamicNN object.

        Args:
            config (list[dict]): List of layer configurations. Each configuration is a dictionary
                specifying the layer type and its corresponding parameters.
            custom_layers (list, optional): List of custom layer classes. Defaults to None.
        """
        # We want to use nn instead of nn.functional for compatibility with Captum
        # For this reason we force cases, i.e. relu will pick up the nn.ReLU class
        available_modules = {
            classname.casefold(): getattr(nn, classname)
            for classname in dir(nn)
            if isinstance(getattr(nn, classname), type)
        }
        if custom_layers is not None:
            for custom_layer in custom_layers:
                available_modules[custom_layer.__name__.casefold()] = custom_layer

        layers = []
        for layer_config in config:
            layer_type = layer_config.pop("type").casefold()
            layer_class = available_modules[layer_type]
            layer_instance = layer_class(**layer_config)
            layers.append(layer_instance)

        super().__init__(*layers)


if __name__ == "__main__":
    from torchvision.ops import MLP
    from xaiunits.model.continuous import ContinuousFeaturesNN

    # Easy quick MLP
    mlp_example = MLP(
        in_channels=10,
        hidden_channels=[64, 128, 256],
        norm_layer=nn.BatchNorm1d,
        activation_layer=nn.ReLU,
        dropout=0.5,
    )
    print(mlp_example)
    random_data = torch.randn(2, 10)
    output = mlp_example(random_data)
    # print(output)

    # Example of using Dynamic NN for a more complex setup
    config = [
        {"type": "Conv2d", "in_channels": 1, "out_channels": 16, "kernel_size": 3},
        {"type": "ReLU"},
        {"type": "MaxPool2d", "kernel_size": 2, "stride": 2},
        {"type": "Flatten"},
        {"type": "Linear", "in_features": 2704, "out_features": 120},
        {"type": "relu"},
        {"type": "Dropout", "p": 0.5},
        {"type": "Linear", "in_features": 120, "out_features": 10},
    ]

    model = DynamicNN(config)
    print(model)
    # Example forward pass
    random_data = torch.randn(32, 1, 28, 28)
    output = model(random_data)
    print(output)

    # Example of using Dynamic NN with custom layers
    weights = torch.tensor([1, 1])
    config = [{"type": "ContinuousFeaturesNN", "n_features": 2, "weights": weights}]
    model = DynamicNN(config, custom_layers=[ContinuousFeaturesNN])
    print(model)
