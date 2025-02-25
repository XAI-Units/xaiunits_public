import pytest
import torch
from xaiunits.model.boolean_not import BooleanNotNN


class TestNOT:
    def test_subclass(self):
        """Test that the model is a subclass of torch.nn.Module"""
        assert issubclass(BooleanNotNN, torch.nn.Module)

    def test_min(self):
        """Test that the model returns negation of the input"""
        model = BooleanNotNN(3)
        res = model(torch.Tensor([99, 199, 3]))
        assert torch.allclose(res, torch.Tensor([-99, -199, -3]))
