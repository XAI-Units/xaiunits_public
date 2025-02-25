import lightning as L
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from typing import Tuple, Any, List, Dict, Union, Optional


class AutoTrainer(L.LightningModule):
    """
    Class that loads up the configuration for performing training, validation, and testing
    torch.nn.Module models using functionality from PyTorch Lightning.

    Requires lightning.Trainer to perform training and testing.

    If more elaborate settings is needed, please add the necessary configurations
    through inheritence, or directly implement as a subclass of lightning.LightningModule.


    Attributes:
        model (torch.nn.Module): The model to be trained.
        loss (function): The loss function used for training.
        optimizer (type): The optimizer used for training.
        optimizer_params (dict): Parameters to be passed to the optimizer.
        test_eval (function): Evaluation function for testing. Defaults to None.
    """

    def __init__(
        self,
        model: Any,
        loss: torch.nn.Module,
        optimizer: Any,
        optimizer_params: Dict = {"lr": 0.0001},
        scheduler: Optional[Any] = None,
        scheduler_params: Optional[Dict] = None,
        test_eval: Optional[Any] = None,
    ) -> None:
        """
        Initializes an AutoTrainer object.

        Args:
            model (torch.nn.Module): The model to be trained.
            loss (function): The loss function used for training.
            optimizer (type): The class of the optimizer used for training.
            optimizer_params (dict): Parameters to be passed to the optimizer.
                Defaults to {"lr": 0.0001}.
            test_eval (function, optional): Evaluation function for testing. Defaults to None.
        """
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.test_eval = test_eval if test_eval is not None else loss
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """
        Performs a forward pass of the model on a batch of training data
        and returns corresponding loss.

        The computed loss is logged as "train_loss".

        Args:
            batch (torch.Tensor): The batch used for training.
            batch_idx (int): Id corresponding to the used batch.

        Returns:
            torch.Tensor: training loss from forward pass of batch into model.
        """
        pred_y = self.model(batch[0])
        loss = self.loss(pred_y.squeeze(), batch[1])
        self.log("train_loss", loss)

        sch = self.lr_schedulers()
        if (
            sch and self.trainer.is_last_batch
        ):  # and (self.trainer.current_epoch + 1) % 1 == 0:
            sch.step()

        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> None:
        """
        Performs a forward pass of the model on a batch of validation data.

        The computed loss is logged as "val_loss".

        Args:
            batch (torch.Tensor): The batch used for training.
            batch_idx (int): Id corresponding to the used batch.
        """
        pred_y = self.model(batch[0])
        val_loss = self.loss(pred_y.squeeze(), batch[1])
        self.log("val_loss", val_loss)

    def test_step(self, batch: Tuple, batch_idx: int) -> None:
        """
        Performs a forward pass of the model on a batch of testing data.

        The computed loss is logged as "test_loss".

        Args:
            batch (torch.Tensor): The batch used for training.
            batch_idx (int): Id corresponding to the used batch.
        """
        pred_y = self.model(batch[0])
        test_eval = self.test_eval(pred_y.squeeze(), batch[1])
        self.log("test_loss", test_eval)

    def forward(self, inputs: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """Perform one forward pass of the model.

        Args:
            inputs (torch.Tensor): Input to the model.

        Returns:
            torch.Tensor: Model output.
        """
        return self.model(inputs)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Define the optimizer of the project.

        Returns:
            Dict: The dictionary containing optimizer and scheduler used for training.
        """
        optimizer = self.optimizer(self.model.parameters(), **self.optimizer_params)
        return_config = {
            "optimizer": optimizer,
        }

        if self.scheduler is not None:
            scheduler = self.scheduler(
                optimizer, **(self.scheduler_params if self.scheduler_params else {})
            )
            return_config["lr_scheduler"] = {"scheduler": scheduler}

        return return_config


if __name__ == "__main__":

    # testing the class
    import numpy as np
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import ToTensor

    # preset dataset on Fashion MNIST
    # training_data = datasets.FashionMNIST(
    #     root="data", train=True, download=True, transform=ToTensor()
    # )
    # testing_data = datasets.FashionMNIST(
    #     root="data", train=False, download=True, transform=ToTensor()
    # )
    # train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)
    # model_config = [
    #     {"type": "Flatten"},
    #     {"type": "Linear", "in_features": 784, "out_features": 10},
    #     {"type": "Softmax", "dim": 1},
    # ]
    # model = DynamicNN(config=model_config)
    # loss = torch.nn.functional.nll_loss
    # optim = torch.optim.Adam
    # lightning_model = AutoTrainer(model, loss, optim)
    # trainer = L.Trainer(
    #     min_epochs=5,
    #     max_epochs=10,
    #     callbacks=[EarlyStopping(monitor="val_loss", mode="min", verbose=True)],
    # )
    # trainer.fit(
    #     model=lightning_model,
    #     train_dataloaders=train_dataloader,
    #     val_dataloaders=test_dataloader,
    # )
    # training with data generated by our data generator
    from xaiunits.datagenerator.data_generation import ConflictingDataset
    from xaiunits.model.dynamic import DynamicNN

    # generate data
    data_gen = ConflictingDataset(
        n_samples=10000, n_features=10, cancellation_likelihood=0.0
    )
    N = len(data_gen.samples)

    # split data
    val_split = N // 10 * 6
    test_split = N // 10 * 8
    train_x, val_x, test_x = (
        data_gen.samples[:val_split],
        data_gen.samples[val_split:test_split],
        data_gen.samples[test_split:],
    )
    train_y, val_y, test_y = (
        data_gen.labels[:val_split],
        data_gen.labels[val_split:test_split],
        data_gen.labels[test_split:],
    )

    # add noise to output labels
    for y in [train_y, val_y, test_y]:
        for i in range(len(y)):
            y[i] += np.random.normal(0, 0.2)  # random noise from N(0, 1)

    # transform to object that is compatible to DataLoader
    train_xy = [(train_x[i], torch.Tensor([train_y[i]])) for i in range(len(train_x))]
    val_xy = [(val_x[i], torch.Tensor([val_y[i]])) for i in range(len(val_x))]
    test_xy = [(test_x[i], torch.Tensor([test_y[i]])) for i in range(len(test_x))]

    # wrap data with DataLoader class
    train_data = DataLoader(train_xy, batch_size=64)
    val_data = DataLoader(val_xy, batch_size=64)
    test_data = DataLoader(test_xy, batch_size=64)

    # define model architecture to define DynamicNN
    n_features = len(data_gen.samples[0])
    linear_model_config = [
        {"type": "Linear", "in_features": n_features, "out_features": 32},
        {"type": "ReLU"},
        {"type": "Linear", "in_features": 32, "out_features": 32},
        {"type": "ReLU"},
        {"type": "Linear", "in_features": 32, "out_features": 32},
        {"type": "ReLU"},
        {"type": "Linear", "in_features": 32, "out_features": 1},
    ]
    linear_model = DynamicNN(linear_model_config)

    # define auto trainer
    loss = torch.nn.functional.mse_loss
    optim = torch.optim.Adam
    lightning_linear_model = AutoTrainer(linear_model, loss, optim)
    trainer = L.Trainer(
        min_epochs=20,
        max_epochs=50,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", verbose=True)],
    )

    # test results before training
    trainer.test(lightning_linear_model, dataloaders=test_data)

    # train model
    trainer.fit(
        model=lightning_linear_model,
        train_dataloaders=train_data,
        val_dataloaders=val_data,
    )

    # test results after training
    trainer.test(lightning_linear_model, dataloaders=test_data)
