import pytest
import torch
from xaiunits.model.boolean_or import BooleanOrNN


class TestOR:
    def test_subclass(self):
        """Test that the model is a subclass of torch.nn.Module"""
        assert issubclass(BooleanOrNN, torch.nn.Module)

    def test_true(self):
        """Test the model returns true"""
        model = BooleanOrNN(3)
        res = model(torch.Tensor([1, -1, -1]))
        assert int(res) == 1

    def test_false(self):
        """Test the model returns false"""
        model = BooleanOrNN(3)
        res = model(torch.Tensor([-1, -1, -1]))
        assert int(res) == -1


# this block is only executed when you run the file directly as `python this_script.py`
# but you should be running `pytest` or `pytest this_script.py`
if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
