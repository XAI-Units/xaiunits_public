import pytest
import torch
import lightning as L
from xaiunits.trainer import AutoTrainer
from xaiunits.model.dynamic import DynamicNN


@pytest.fixture
def trainer():
    model_config = [
        {"type": "Linear", "in_features": 16, "out_features": 8},
        {"type": "ReLU"},
        {"type": "Linear", "in_features": 8, "out_features": 4},
        {"type": "ReLU"},
        {"type": "Linear", "in_features": 4, "out_features": 2},
        {"type": "ReLU"},
        {"type": "Linear", "in_features": 2, "out_features": 1},
    ]
    model = DynamicNN(model_config)
    loss = torch.nn.functional.mse_loss
    optimizer = torch.optim.Adam
    optimizer_params = {"lr": 0.01}
    test_eval = torch.nn.functional.l1_loss
    return AutoTrainer(model, loss, optimizer, optimizer_params, test_eval)


@pytest.fixture
def batch_input():
    return torch.randn((2, 16))


class TestAutoTrainer:
    def test_init(self, trainer):
        """Tests the __init__() method of AutoTrainer."""
        model_config = [
            {"type": "Linear", "in_features": 16, "out_features": 8},
            {"type": "ReLU"},
            {"type": "Linear", "in_features": 8, "out_features": 4},
            {"type": "ReLU"},
            {"type": "Linear", "in_features": 4, "out_features": 2},
            {"type": "ReLU"},
            {"type": "Linear", "in_features": 2, "out_features": 1},
        ]
        model = DynamicNN(model_config)
        loss = torch.nn.functional.mse_loss
        optimizer = torch.optim.Adam
        optimizer_params = {"lr": 0.01}
        test_eval = torch.nn.functional.l1_loss
        trainer = AutoTrainer(model, loss, optimizer, optimizer_params, test_eval)

        assert trainer.model == model
        assert trainer.loss == loss
        assert trainer.optimizer == optimizer
        assert trainer.optimizer_params == optimizer_params
        assert trainer.test_eval == test_eval

    def test_subclass(self):
        """
        Tests the AutoTrainer class is a subclass of lightning.LightningModule.
        """
        assert issubclass(AutoTrainer, L.LightningModule)

    def test_training_step(self, trainer, batch_input):
        """Tests the training_step() method of AutoTrainer."""
        batch_output = trainer.model(batch_input)
        batch = (batch_input, batch_output.squeeze())
        nonzero_loss = torch.all(trainer.training_step(batch, 0) == 0)
        assert nonzero_loss.item()

    def test_forward(self, trainer, batch_input):
        """Tests the forward() method of AutoTrainer."""
        model_output = trainer.model(batch_input)
        output = trainer.forward(batch_input)
        assert torch.all(output == model_output).item()

    def test_configure_optimizers(self, trainer):
        """Tests the configure_optimizers() method of AutoTrainer."""
        optimizer = trainer.configure_optimizers()
        assert isinstance(optimizer, trainer.optimizer)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
