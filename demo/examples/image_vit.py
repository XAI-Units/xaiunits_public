# %%
import copy
from safetensors.torch import save_file
from safetensors import safe_open
import lightning as L
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms
from captum.attr import *
from torch.utils.data import DataLoader
from xaiunits.datagenerator import *
from xaiunits.pipeline import Experiment, ExperimentPipeline, Results
from vison_transformer import VisionTransformer, train_vit


training = False

datasets = [
    BalancedImageDataset,
    ImbalancedImageDataset,
]

geo_shapes = [
    "ellipse",
    "nonagon",
    "hexagon",
    "circle",
    "heptagon",
    "octagon",
    "decagon",
    "rectangle",
    "triangle",
    "pentagon",
]
dino_shapes = [
    "Carcharodontosaurus_UDL",
    "Shantungosaurus_giganteus",
    "Life_reconstruction_of_Tlatolophus_galorum",
    "Alxasaurus_UDL",  # UDL
    "Aletopelta_UDL",  # UDL
    "Caudipteryx_UDL",
    "Imperobator_UDL",  # UDL
    "Jiangjunosaurus_junggarensis",
    "Nanuqsaurus_UDL",
    "Stegosaurus.svg",
]


base_params = {
    "transform": transforms.Compose(
        [transforms.ToTensor(), transforms.ToDtype(torch.float32, scale=True)]
    ),
    "background_size": (512, 512),
    "position": "random",
    "overlay_scale": 0.4,
    "contour_thickness": 100,
    "backgrounds": [
        "smeared_0116.jpg",
        "marbled_0185.jpg",
        "frilly_0065.jpg",
        "swirly_0123.jpg",
        "crosshatched_0127.jpg",
        "meshed_0180.jpg",
        "waffled_0113.jpg",
        "zigzagged_0034.jpg",
        "sprinkled_0121.jpg",
        "braided_0177.jpg",
    ],
    "source": "url",
}
datasets_param = [
    # Train
    base_params
    | {
        "shapes": dino_shapes,
        "shape_type": "dinosaurs",
        "n_variants": 24,
        "shape_colors": ["blue"],
    },  # DINO_BALANCE,
    base_params
    | {
        "shapes": dino_shapes,
        "shape_type": "dinosaurs",
        "n_variants": 240,
        "shape_colors": "blue",
    },  # DINO_IMBALANCE,
    # Test
    base_params
    | {
        "shapes": dino_shapes,
        "shape_type": "dinosaurs",
        "n_variants": 3,
        "shape_colors": ["blue"],
    },  # DINO_BALANCE,
    base_params
    | {
        "shapes": dino_shapes,
        "shape_type": "dinosaurs",
        "n_variants": 30,
        "shape_colors": "blue",
    },  # DINO_IMBALANCE,
    # Val
    base_params
    | {
        "shapes": dino_shapes,
        "shape_type": "dinosaurs",
        "n_variants": 3,
        "shape_colors": ["blue"],
    },  # DINO_BALANCE,
    base_params
    | {
        "shapes": dino_shapes,
        "shape_type": "dinosaurs",
        "n_variants": 30,
        "shape_colors": "blue",
    },  # DINO_IMBALANCE,
]


methods = [DeepLift, IntegratedGradients, InputXGradient]

# Hyperparams
lr = 0.00005
optim = torch.optim.Adam
batch_size = 16
accumulate_grad_batches = 1
loss = torch.nn.functional.cross_entropy
vit_config = {
    "d_model": 128,
    "n_classes": 10,
    "img_size": (512, 512),
    "patch_size": (32, 32),
    "n_channels": 3,
    "n_heads": 2,
    "n_layers": 8,
}
version = "v2"
# %%
if __name__ == "__main__":
    results = Results()
    test_eval_scores = []
    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        for i, data in enumerate(datasets):
            inter = []
            for j in range(3):
                inter.append(
                    data(
                        **(
                            datasets_param[i + j * len(datasets)]
                            | {"seed": seed * 100 + i * 20 + j}
                        )
                    )
                )

            train_dataset, test_dataset, val_dataset = inter
            fg_type = "dino"
            test_dataset.name = f"{test_dataset.__class__.__name__}_{fg_type}"

            (train_dl,) = [
                DataLoader(x, batch_size=batch_size, shuffle=True, num_workers=4)
                for x in [train_dataset]
            ]
            val_dl, test_dl = [
                DataLoader(x, batch_size=batch_size, shuffle=False, num_workers=4)
                for x in [val_dataset, test_dataset]
            ]

            torch.manual_seed(seed)

            if training:
                trained_model = train_vit(
                    test_dataset.name + "_" + str(seed) + "_" + str(version),
                    train_dl,
                    val_dl,
                    test_dl,
                    seed=seed * 100 + i * 20 + j,
                    vit_config=vit_config,
                    lr=lr,
                    accumulate_grad_batches=accumulate_grad_batches,
                )
            else:
                # Load Trained Model
                trained_model = VisionTransformer.from_pretrained(
                    f"trained_models/vit/{test_dataset.name}_{seed}_{version}"
                )
                # Pre-Trained model can be loaded from hugging face website.

            trained_model.eval()
            total_params = sum(p.numel() for p in trained_model.parameters())
            print(f"Number of parameters: {total_params}")

            model_out = []
            for batch in test_dl:

                output = trained_model(batch[0])
                output = torch.argmax(output.squeeze(), dim=1)
                model_out.append(output.squeeze())

            model_out = torch.concat(model_out, dim=0)

            accuracy = (
                (model_out == torch.tensor(test_dataset[:][1])).float().sum()
            ) / len(test_dataset)
            test_eval_scores.append(accuracy)

            ExperimentPipeline(
                Experiment(
                    test_dataset,
                    trained_model.cpu(),
                    methods,
                    None,
                    # method_seeds=[0],
                    name=f"data_model_seed_{seed}",
                ),
                batch_size=1,
                results=results,
                default_target="predicted_class",
            ).run()

    # %%
    # Print Eval Test Score
    new = torch.stack(test_eval_scores)
    new = torch.stack(
        [
            new[i : i + len(datasets)]
            for i in range(0, len(datasets) * (len(seeds)), len(datasets))
        ]
    )
    mean = new.mean(dim=0)
    std = new.std(dim=0)
    final = []
    for m, s in zip(mean, std):
        final.append(f"{m:.3f}  ± {s:.3f}")
    print(" & ".join(final))

    # %%

    latex = False
    df = results.data

    df["foreground_image"] = df["data"].apply(
        lambda x: "Dinosaurs" if x.split("_")[1] == "dino" else "Geometric"
    )
    df["base_dataset"] = df["data"].apply(
        lambda x: (
            "BalancedImageDataset"
            if x.split("_")[0] == "BalancedImageDataset"
            else "ImbalancedImageDataset"
        )
    )
    df["value_adj"] = df["value"].apply(lambda x: x)

    df.method = pd.Categorical(
        df.method, categories=["DeepLift", "InputXGradient", "IntegratedGradients"]
    )
    # print(df)

    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.max_columns = 100
    df = pd.pivot_table(
        df,
        values=["value_adj"],
        index=["base_dataset", "foreground_image", "method", "trial_group_name"],
    )
    # print(df)
    mu = pd.pivot_table(
        df,
        index=["method"],
        values=["value_adj"],
        columns=["base_dataset"],
        aggfunc="mean",
    )

    std = pd.pivot_table(
        df,
        index=["method"],
        values=["value_adj"],
        columns=["base_dataset"],
        aggfunc="std",
    )

    combined = pd.pivot_table(
        pd.concat([mu, std], axis=0),
        index=["method"],
        values=["value_adj"],
        aggfunc=[lambda col: " ± ".join([f"{r:.3f}" for r in col])],
    )

    if latex:
        combined = combined.to_latex()

    print(combined)
