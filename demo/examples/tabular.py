# %%
import copy
import os

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr import *
from captum.metrics import infidelity, sensitivity_max
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from xaiunits.datagenerator import *
from xaiunits.methods import wrap_method
from xaiunits.metrics import wrap_metric
from xaiunits.model import DynamicNN
from xaiunits.pipeline import Experiment, ExperimentPipeline, Pipeline, Results
from xaiunits.trainer.trainer import AutoTrainer

if __name__ == "__main__":
    # %%
    # Training and Running Experiment
    training = True

    datasets = [
        WeightedFeaturesDataset,
        ConflictingDataset,
        InteractingFeatureDataset,
        UncertaintyAwareDataset,
        ShatteredGradientsDataset,
        PertinentNegativesDataset,
        BooleanAndDataset,
        BooleanOrDataset,
    ]

    base_params = {"n_features": 10, "n_samples": 1000 * 4}
    datasets_param = [
        base_params,  # WeightedFeaturesDataset,
        base_params,  # ConflictingDataset,
        base_params
        | {
            "interacting_features": [[i, i + 1] for i in range(0, 10, 2)],
        },  # InteractingFeatureDataset,
        base_params
        | {
            "common_features": 5,
        },  # UncertaintyAwareDataset,
        base_params | {"weight_scale": 50},  # ShatteredGradientsDataset,
        base_params
        | {
            "pn_features": list(range(5)),
        },  # PertinentNegativesDataset
        base_params,  # BooleanAndDataset,
        base_params,  # BooleanOrDataset,
    ]

    methods = [
        DeepLift,
        Lime,
        wrap_method(
            Lime,
            class_params={"interpretable_model": SkLearnLinearRegression()},
            pre_fix="Simple_Linear_",
        ),
        KernelShap,
        ShapleyValueSampling,
        IntegratedGradients,
        InputXGradient,
    ]

    lr = 0.00001
    optim = torch.optim.Adam

    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight.data)

    results = Results()
    test_eval_scores = []
    seeds = [0, 1, 2, 3, 4]
    for data, params in zip(datasets, datasets_param):
        # results = Results()
        for seed in seeds:
            if data.__name__ == "UncertaintyAwareDataset":
                loss = torch.nn.functional.cross_entropy
                test_eval = (
                    lambda x, y: (torch.argmax(x, dim=1).squeeze() == y).sum()
                    / x.shape[0]
                )
                out_dim = 5
                default_target = "predicted_class"
            else:
                loss = torch.nn.functional.mse_loss
                test_eval = None
                out_dim = 1
                default_target = None

            instance = data(**(params | {"seed": seed}))
            model = instance.generate_model()

            test_dataset, train_dataset, val_dataset = instance.split(
                [0.25, 0.65, 0.10]
            )

            train_dl, val_dl, test_dl = [
                DataLoader(x, batch_size=256, shuffle=False, num_workers=4)
                for x in [train_dataset, val_dataset, test_dataset]
            ]

            hdim = 100
            linear_model_config = [
                {
                    "type": "Linear",
                    "in_features": instance[:][0].shape[1],
                    "out_features": hdim,
                },
                {"type": "ReLU"},
                {"type": "Linear", "in_features": hdim, "out_features": hdim},
                {"type": "ReLU"},
                {"type": "Linear", "in_features": hdim, "out_features": hdim},
                {"type": "ReLU"},
                {"type": "Linear", "in_features": hdim, "out_features": hdim},
                {"type": "ReLU"},
                {"type": "Linear", "in_features": hdim, "out_features": out_dim},
            ]

            torch.manual_seed(seed)
            trained_model = DynamicNN(linear_model_config)

            if training:
                # Train Model
                trainer = L.Trainer(
                    min_epochs=100,
                    # max_epochs=1000,
                    max_time="00:00:30:00",
                    callbacks=[
                        EarlyStopping(monitor="val_loss", mode="min", verbose=False)
                    ],
                )
                llm = AutoTrainer(
                    trained_model,
                    loss,
                    optim,
                    optimizer_params={"lr": lr},
                    test_eval=test_eval,
                )
                trainer.fit(
                    model=llm,
                    train_dataloaders=train_dl,
                    val_dataloaders=val_dl,
                )
                trainer.test(llm, dataloaders=test_dl)
                trained_model = llm.model

                current_directory = os.getcwd()
                final_directory = os.path.join(current_directory, "trained_models")
                if not os.path.exists(final_directory):
                    os.makedirs(final_directory)
                save_file(
                    trained_model.cpu().state_dict(),
                    f"trained_models/{data.__name__}_{seed}.pt",
                )

            else:
                # Load trained model
                tensors = {}
                with safe_open(
                    f"trained_models/{data.__name__}_{seed}.pt",
                    framework="pt",
                    device="cpu",
                ) as f:
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key)
                trained_model.load_state_dict(tensors)

            if data.__name__ == "UncertaintyAwareDataset":
                test_eval_score = (
                    torch.argmax(
                        trained_model(test_dataset[:][0]).squeeze(), dim=1
                    ).squeeze()
                    == test_dataset[:][1]
                ).sum() / len(test_dataset)
            else:
                test_eval_score = torch.nn.functional.mse_loss(
                    trained_model(test_dataset[:][0]).squeeze(), test_dataset[:][1]
                )
            test_eval_scores.append(test_eval_score)

            ExperimentPipeline(
                Experiment(
                    test_dataset,
                    [model, trained_model],
                    methods,
                    metrics=None,
                    # method_seeds=[0],
                    name=f"data_model_seed_{seed}",
                ),
                batch_size=1000,
                results=results,
                default_target=default_target,
            ).run()
    # %%
    # Print Eval Test Score
    new = torch.stack(test_eval_scores)
    new = torch.stack(
        [
            new[i : i + len(datasets)]
            for i in range(0, len(datasets) * len(seeds), len(datasets))
        ]
    )
    mean = new.mean(dim=0)
    std = new.std(dim=0)
    final = []
    for m, s in zip(mean, std):
        final.append(f"{m:.3f}  ± {s:.3f}")
    print(" & ".join(final))

    # %%
    # Save results
    try:
        results.print_stats()
    except:
        pass
    df = results.data
    df.to_csv(f"results_all.csv", sep=",", index=False, encoding="utf-8")

    # %%
    # Print Experiment Results
    mapping = {
        "WeightedFeaturesDataset": "Wtg Fts\\tnote{1}",
        "ConflictingDataset": "Conflicting\\tnote{2}",
        "InteractingFeatureDataset": "Interacting\\tnote{3}",
        "UncertaintyAwareDataset": "Uncertainty\\tnote{4}",
        "ShatteredGradientsDataset": "Shattered Grad\\tnote{5}",
        "PertinentNegativesDataset": "Pertinent Neg\\tnote{6}",
        "BooleanAndDataset": "Bool AND\\tnote{7}",
        "BooleanOrDataset": "Bool OR\\tnote{8}",
        "Subset": "Z",
    }

    def model_adj(row):
        if row["model"] == "DynamicNN":
            return "Trained"
        else:
            return "Handcrafted"

    def data_adj(row):
        return mapping.get(row["data"], row["data"])

    for half in [list(mapping)[:4], list(mapping)[4:]]:
        df = results.data
        df = df.loc[df["data"].isin(half), :]
        df.loc[df["method"] == "Lime", "method"] = "Lime (Lasso)"
        df.loc[df["method"] == "Simple_Linear_Lime", "method"] = "Lime (Linear)"
        df["model"] = df.apply(model_adj, axis=1, raw=False, result_type="expand")
        df["data"] = df.apply(data_adj, axis=1, raw=False, result_type="expand")
        df.model = pd.Categorical(df.model, categories=["Handcrafted", "Trained"])
        df.method = pd.Categorical(
            df.method,
            categories=[
                "DeepLift",
                "InputXGradient",
                "IntegratedGradients",
                "KernelShap",
                "ShapleyValueSampling",
                "Lime (Linear)",
                "Lime (Lasso)",
            ],
        )
        df.data = pd.Categorical(df.data, categories=list(mapping.values()))
        df = df.sort_values("model")

        pd.options.display.float_format = "{:,.3f}".format
        pd.options.display.max_columns = 100
        df = pd.pivot_table(
            df, values=["value"], index=["method", "model", "data", "trial_group_name"]
        )

        mu = pd.pivot_table(
            df,
            index=["method", "model"],
            values=["value"],
            columns=["data"],
            aggfunc="mean",
        )

        std = pd.pivot_table(
            df,
            index=["method", "model"],
            values=["value"],
            columns=["data"],
            aggfunc="std",
        )

        combined = pd.pivot_table(
            pd.concat([mu, std], axis=0),
            index=["method", "model"],
            values=["value"],
            aggfunc=[lambda col: " ± ".join([f"{r:.3f}" for r in col])],
        )

        print(combined)
