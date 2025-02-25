import pickle
import numpy as np
import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.transforms.v2 as transforms
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from xaiunits.trainer.trainer import AutoTrainer
from huggingface_hub import PyTorchModelHubMixin
from torch.utils.data import DataLoader


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()

        self.d_model = d_model  # Dimensionality of Model
        self.img_size = img_size  # Image Size
        self.patch_size = patch_size  # Patch Size
        self.n_channels = n_channels  # Number of Channels

        self.linear_project = nn.Conv2d(
            self.n_channels,
            self.d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        x = self.linear_project(x)

        x = x.flatten(2)

        x = x.transpose(1, 2)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Creating positional encoding
        pe = torch.zeros(max_seq_length, d_model)

        for pos in range(max_seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos / (10000 ** (i / d_model)))
                else:
                    pe[pos][i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)
        x = torch.cat((tokens_batch, x), dim=1)
        x = x + self.pe

        return x


class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Obtaining Queries, Keys, and Values
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Dot Product of Queries and Keys
        attention = Q @ K.transpose(-2, -1)

        attention = attention / (self.head_size**0.5)

        attention = torch.softmax(attention, dim=-1)

        attention = self.dropout(attention)

        attention = attention @ V

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.head_size = d_model // n_heads

        self.W_o = nn.Linear(d_model, d_model)

        self.heads = nn.ModuleList(
            [AttentionHead(d_model, self.head_size) for _ in range(n_heads)]
        )

    def forward(self, x):
        # Combine attention heads
        out = torch.cat([head(x) for head in self.heads], dim=-1)

        out = self.W_o(out)

        return out


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.ln1 = nn.LayerNorm(d_model)

        self.mha = MultiHeadAttention(d_model, n_heads)

        self.ln2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * r_mlp),
            nn.GELU(),
            nn.Linear(d_model * r_mlp, d_model),
        )

    def forward(self, x):
        # Residual Connection After Sub-Layer 1
        out = x + self.mha(self.ln1(x))

        # Residual Connection After Sub-Layer 2
        out = out + self.mlp(self.ln2(out))

        return out


class VisionTransformer(
    nn.Module,
    PyTorchModelHubMixin,
    # repo_url="your-repo-url",
    # pipeline_tag="text-to-image",
    # license="mit",
):
    def __init__(
        self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers
    ):
        super().__init__()

        assert (
            img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0
        ), "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model  # Dimensionality of model
        self.n_classes = n_classes  # Number of classes
        self.img_size = img_size  # Image size
        self.patch_size = patch_size  # Patch size
        self.n_channels = n_channels  # Number of channels
        self.n_heads = n_heads  # Number of attention heads

        # Calculate number of patches
        self.n_patches = (self.img_size[0] * self.img_size[1]) // (
            self.patch_size[0] * self.patch_size[1]
        )
        self.max_seq_length = self.n_patches + 1  # Add 1 for class token

        # Components of the Vision Transformer
        self.patch_embedding = PatchEmbedding(
            self.d_model, self.img_size, self.patch_size, self.n_channels
        )
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_length)
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoder(self.d_model, self.n_heads) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(0.1)
        # Classification MLP (without Softmax)
        self.classifier = nn.Linear(self.d_model, self.n_classes)

    def forward(self, images):
        x = self.patch_embedding(images)  # Get patch embeddings
        x = self.positional_encoding(x)  # Add positional encoding
        x = self.transformer_encoder(x)  # Apply Transformer encoders
        x = self.dropout(x[:, 0])  # Apply dropout
        x = self.classifier(x)  # Use the class token for classification
        return x


def train_vit(
    model_name,
    train_dl,
    val_dl,
    test_dl,
    seed,
    lr,
    vit_config,
    accumulate_grad_batches=1,
    save=True,
):
    torch.manual_seed(seed)
    model = VisionTransformer(**vit_config).float()

    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # define auto trainer
    loss = torch.nn.functional.cross_entropy
    optim = torch.optim.Adam

    lightning_linear_model = AutoTrainer(
        model,
        loss,
        optim,
        test_eval=lambda x, y: (torch.argmax(x, dim=1).squeeze() == y).sum()
        / x.shape[0],
        optimizer_params={"lr": lr},
    )

    trainer = L.Trainer(
        min_epochs=30,
        # max_epochs=1000,
        # max_time="00:00:30:00",
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="max", verbose=False, patience=3)
        ],
        accumulate_grad_batches=accumulate_grad_batches,
    )

    # test results before training
    trainer.test(lightning_linear_model, dataloaders=test_dl)

    # train model
    trainer.fit(
        model=lightning_linear_model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
    )
    # test results after training
    trainer.test(lightning_linear_model, dataloaders=test_dl)

    model = lightning_linear_model.model

    if save:
        model.save_pretrained("trained_models/vit/" + model_name)

    return model
