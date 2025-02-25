import torch
import torch.nn as nn


class BooleanNotNN(nn.Sequential):
    """
    Implements a neural network model designed to apply the logical NOT operation to input features.

    This model consists of a single linear layer without bias, configured to negate each input feature.
    The weights of the layer are initialized to -1 for each feature, effectively performing the 'NOT' operation
    in a bitwise manner when the inputs are considered to be -1 or 1.

    Inherits from:
        torch.nn.Sequential: Parent class for implementing neural networks with modules defined in a
        sequential manner.
    """

    def __init__(self, n_features: int) -> None:
        """
        Initializes a NOT model with a specified dimension for the input features.

        The initialized model consists of a single linear layer without bias. The weights of this layer are
        set to -1 for each input feature, enabling the layer to negate the values of all input features.

        Args:
            n_features (int): The total number of features in the input data. This parameter determines the
                dimensionality of the input to the linear layer as well as the output dimension, allowing the
                model to apply the NOT operation to each feature independently.
        """
        N = n_features
        l0 = nn.Linear(N, N, bias=False)

        # initialize weights for l0
        w0 = torch.eye(N) * -1
        l0.weight = nn.Parameter(w0.float())

        super().__init__(l0)


if __name__ == "__main__":
    model = BooleanNotNN(3)
    res = model(torch.Tensor([99, 199934, 3]))
    print(res)
