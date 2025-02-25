# %%
import copy

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr import *
from captum.metrics import infidelity, sensitivity_max
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from xaiunits.datagenerator import *
from xaiunits.methods import wrap_method
from xaiunits.metrics import wrap_metric
from xaiunits.model import DynamicNN
from xaiunits.pipeline import Experiment, ExperimentPipeline, Pipeline, Results
from xaiunits.trainer.trainer import AutoTrainer


class DeepLiftNormalised:
    def __init__(
        self,
        model,
    ):
        from captum.attr import DeepLift

        self.attr = DeepLift(model)

    def attribute(self, inputs, target, baselines, output_dim=5, **other_args):
        output_class = []
        for i in range(output_dim):
            output = self.attr.attribute(inputs, baselines, i, **other_args)
            output_class.append(output)

        output_target = self.attr.attribute(inputs, baselines, target, **other_args)
        output_target = output_target - torch.stack(output_class, dim=0).mean(dim=0)

        return output_target


if __name__ == "__main__":
    test = []
    for idx in [0, 1, 2, 3, 4]:
        instance = UncertaintyAwareDataset(
            **{
                "n_features": 10,
                "n_samples": 1000 * 4,
                "common_features": 5,
                "seed": idx,
            }
        )
        test_dataset, train_dataset, val_dataset = instance.split([0.25, 0.65, 0.10])

        hmodel_soft = instance.generate_model(softmax_layer=True)
        hmodel_soft.model_name = "HandCrafted Model with SoftMax"
        hmodel_no_soft = instance.generate_model(softmax_layer=False)
        hmodel_no_soft.model_name = "HandCrafted Model without SoftMax"

        combo = [
            [hmodel_soft, DeepLift, test_dataset],
            [hmodel_no_soft, DeepLift, test_dataset],
            [hmodel_no_soft, DeepLiftNormalised, test_dataset],
        ]
        for model, method, data in combo:
            res = torch.argmax(model(data[:][0]).squeeze(), dim=1)
            dl = method(model)
            attribute = dl.attribute(
                data[:][0], target=res, baselines=torch.zeros_like(data[:][0])
            )
            #

            base_data = {
                "seed": idx,
                "model": model.model_name,
                "method": method.__name__,
                "data": "Narrow" if data is test_dataset else "wide",
            }

            if model.model_name.endswith("with SoftMax"):
                base_data["softmax"] = "True"
            else:
                base_data["softmax"] = "False"

            base_data["base model"] = "Handcrafted"

            mse = (
                torch.nn.functional.mse_loss(
                    attribute, data[:][2]["ground_truth_attribute"], reduction="none"
                )[:, list(range(-1, -5 - 1, -1))]
                .mean(dim=1)
                .detach()
            )
            test.append(
                base_data
                | {
                    "metric": "masked mse",
                    "mean_data": mse.mean(dim=0).item(),
                    "mean_std": mse.std(dim=0).item(),
                }
            )

    df = pd.DataFrame(test)
    df.softmax = pd.Categorical(df.softmax, categories=["True", "False"])

    # print(df)
    # pd.set_option("display.precision", 5)
    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.max_columns = 100

    index = ["base model"]
    columns = ["method", "softmax"]
    values = ["mean_data"]
    df = pd.pivot_table(df, values=values, index=columns + index + ["seed"])

    mu = pd.pivot_table(df, index=index, values=values, columns=columns, aggfunc="mean")
    std = pd.pivot_table(df, index=index, values=values, columns=columns, aggfunc="std")
    print(mu, std)
    combined = pd.pivot_table(
        pd.concat([mu, std], axis=0),
        index=index,
        values=values,
        aggfunc=[lambda col: " ± ".join([f"{r:.3f}" for r in col])],
    )

    print(combined)
