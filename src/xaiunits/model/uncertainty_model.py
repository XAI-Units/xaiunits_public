import torch
import torch.nn as nn
from xaiunits.model import generate_layers


class UncertaintyNN(nn.Sequential):
    """
    Implements a neural network model designed to capture behaviour were an input node impact all or several output nodes equally.

    This model uses a linear layer followed by a softmax activation function to compute a distribution over the
    input features, effectively capturing the uncertainty or confidence level associated with each feature.

    For best practice, please create an instance of network using generate_model() method of UncertaintyAwareDataset.

    Inherits from:
        torch.nn.Sequential: Parent class for implementing neural networks with modules defined in a
        sequential manner.
    """

    def __init__(
        self, n_features: int, weights: torch.Tensor, softmax_layer: bool = True
    ) -> None:
        """
        Initializes the UncertaintyNN model with a specified dimension for the input features and custom weights.

        The initialized model consists of a single linear layer without bias, followed by a softmax layer
        to normalize the output of the linear layer into a probability distribution,
        representing the model's confidence or uncertainty regarding each input feature.

        Args:
            n_features (int): The total number of features in the input data. This parameter determines the
                dimensionality of the input to the linear layer and subsequently the output dimension of the model
                before applying the softmax.
            weights (torch.Tensor): A torch.Tensor object to directly specify the weights for the linear
                transformation of the input features.
        """
        assert weights.shape[1] == n_features
        if softmax_layer:
            act = nn.Softmax(dim=1)
        else:
            act = None
        layers = generate_layers(weights=weights, biases=None, act_fns=act)

        super().__init__(*layers)


if __name__ == "__main__":
    w = torch.Tensor([[1, 1, 1, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float().T
    res = UncertaintyNN(4, weights=w)(torch.Tensor([[1, 0, 0, 0]]))
    print(torch.max(res, dim=1)[1])
    print(type(nn.Softmax))
