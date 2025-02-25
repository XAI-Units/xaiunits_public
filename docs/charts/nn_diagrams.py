import os

import torch
import torch.nn as nn
from graphviz import Digraph
from PIL import Image
from sympy import symbols
from torch import nn
from xaiunits.datagenerator import (
    BooleanDataset,
    ConflictingDataset,
    # ImageDataset,
    InteractingFeatureDataset,
    PertinentNegativesDataset,
    ShatteredGradientsDataset,
    UncertaintyAwareDataset,
    WeightedFeaturesDataset,
)
from xaiunits.model import ConflictingFeaturesNN


class ModelVisualizer:
    def __init__(
        self, model, cat_features=[], include_weights=False, dash_threshold=100
    ):
        self.modules = list(model.children())
        self.cat_features = cat_features
        self.include_weights = include_weights
        self.dash_threshold = dash_threshold
        self.dot = Digraph()
        self.fontsize = "18"
        self._graph_settings()

    def _graph_settings(self):
        self.dot.attr(rankdir="LR", splines="line", ranksep="1.0", pad="-0.0")
        self.dot.node_attr.update(
            style="solid",
            shape="circle",
            color="black",
            width="0.4",
            height="0.4",
            margin="0",
            fontsize=self.fontsize,
        )

    def visualize(self, file_path=None, view=True):
        first_layer = self.modules[0]
        assert isinstance(first_layer, nn.Linear), "First layer expected to be Linear"
        previous_layer_outputs = self._add_input_layer(first_layer.in_features)

        all_weights = [
            layer.weight.data.view(-1)
            for layer in self.modules
            if isinstance(layer, nn.Linear)
        ]

        cluster_index = 1
        for layer in self.modules:
            if isinstance(layer, nn.Linear):
                previous_layer_outputs = self._add_layer(
                    layer, previous_layer_outputs, cluster_index
                )
                cluster_index += 1  # only increment if a layer has been added

        self.dot.render(file_path, format="svg", view=view, cleanup=True)
        self.dot.render(file_path, format="png", view=view, cleanup=True)
        return self.dot

    def _add_input_layer(self, input_features):
        with self.dot.subgraph(name="cluster_0") as c:
            c.attr(label="Input", color="white", fontsize=self.fontsize)
            for i in range(input_features):
                node_name = f"input_{i}"
                cat_features = [x % input_features for x in self.cat_features]  # for -1
                shape = "diamond" if i in cat_features else "circle"
                c.node(node_name, label=str(i), shape=shape)
        return [f"input_{i}" for i in range(input_features)]

    def _add_layer(self, layer, previous_layer_outputs, cluster_index):
        next_activation = self._get_next_activation(layer)
        with self.dot.subgraph(name=f"cluster_{cluster_index}") as c:
            label = f"Linear+{next_activation}" if next_activation else "Linear"
            c.attr(label=label, color="white", fontsize=self.fontsize)
            new_layer_outputs = []

            for j in range(layer.out_features):
                node_name = f"node_{cluster_index}_{j}"
                c.node(node_name, label=str(j))
                new_layer_outputs.append(node_name)

                self._add_edges(layer, previous_layer_outputs, new_layer_outputs, j)

        return new_layer_outputs

    def _get_next_activation(self, layer):
        idx = self.modules.index(layer) + 1
        if idx < len(self.modules):
            next_layer = self.modules[idx]
            if isinstance(next_layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softmax)):
                return type(next_layer).__name__
        return None

    def _add_edges(
        self,
        layer,
        previous_layer_outputs,
        new_layer_outputs,
        current_output_index,
    ):
        weight_matrix = layer.weight.data.view(-1).clone()
        max_weight = self._max_valid_weight(weight_matrix)
        for in_index, prev in enumerate(previous_layer_outputs):
            weight = weight_matrix[
                current_output_index * len(previous_layer_outputs) + in_index
            ]

            if weight != 0:
                self.dot.edge(
                    prev,
                    new_layer_outputs[current_output_index],
                    style="dashed" if weight.abs() >= self.dash_threshold else "solid",
                    color="blue" if weight > 0 else "red",
                    arrowhead="none",
                    label=f"{weight:.1f}" if self.include_weights else None,
                )

    def _max_valid_weight(self, weights):
        abs_weights = weights.abs()
        valid_abs_weights = abs_weights[abs_weights < self.dash_threshold]
        return torch.max(valid_abs_weights) if len(valid_abs_weights) > 0 else 0.0


def plot_legend(folder="charts/nn_diagrams"):
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    # Create custom legend handles with no connecting lines for markers
    blue_line = mlines.Line2D([], [], color="blue", label="Positive Weight")
    red_line = mlines.Line2D([], [], color="red", label="Negative Weight")
    dashed_line = mlines.Line2D(
        [], [], linestyle=(0, (5, 5)), color="black", label="Extreme Outlier Weight"
    )
    circle_marker = mlines.Line2D(
        [],
        [],
        marker="o",
        color="black",
        linestyle="None",
        markersize=12,
        label="Continuous Feature",
    )
    diamond_marker = mlines.Line2D(
        [],
        [],
        marker="D",
        color="black",
        linestyle="None",
        markersize=12,
        label="Categorical Feature",
    )

    # Create the figure and axis objects
    fig, ax = plt.subplots()

    # Add the legend to the plot
    legend = ax.legend(
        handles=[blue_line, red_line, dashed_line, circle_marker, diamond_marker],
        loc="center",
        fontsize=18,
        frameon=False,
    )
    ax.axis("off")  # Hide axes

    # Draw the figure to update the renderer and get correct size
    fig.canvas.draw()

    # Get the bbox of the legend and adjust figure size
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.set_size_inches(bbox.width, bbox.height)  # Set the figure size based on legend

    # Adjust the figure padding to be tight around the legend
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.savefig(f"{folder}/legend.svg")
    plt.savefig(f"{folder}/legend.png")


def pad_image_to_match_size(image_path, target_size, output_path):
    # Open the image to pad
    image = Image.open(image_path)
    image_size = image.size

    # Calculate padding needed
    padding_x = max(target_size[0] - image_size[0], 0)
    padding_y = max(target_size[1] - image_size[1], 0)

    # Apply padding, adjusting for any non-even division
    left_padding = padding_x // 2
    right_padding = padding_x - left_padding
    top_padding = padding_y // 2
    bottom_padding = padding_y - top_padding

    # If padding cannot be evenly divided, adjust right and bottom padding
    if padding_x % 2 != 0:
        right_padding += 1
    if padding_y % 2 != 0:
        bottom_padding += 1

    # Create a new image with white background and the original image at the center
    new_image = Image.new(image.mode, target_size, "WHITE")
    new_image.paste(image, (left_padding, top_padding))

    # Save the padded image
    new_image.save(output_path)


def save_all_datasets(folder="charts/nn_diagrams"):
    x, y = symbols("x y")
    datasets = {
        "conflicting": ConflictingDataset(seed=1, n_features=1),
        "conflicting_double": ConflictingDataset(),
        "boolean_formula_not": BooleanDataset(~x),
        "boolean_formula_and": BooleanDataset(x & y),
        "boolean_formula_or": BooleanDataset(x | y),
        "interacting_features": InteractingFeatureDataset(
            n_features=2, interacting_features=[[1, 0]]
        ),
        "pertinent_negatives": PertinentNegativesDataset(n_features=2),
        "shattered_gradients": ShatteredGradientsDataset(n_features=3),
        "uncertainty_aware": UncertaintyAwareDataset(n_features=4),
        "weighted_features": WeightedFeaturesDataset(n_features=1),
    }
    for name, dataset in datasets.items():
        model = dataset.generate_model()
        visualizer = ModelVisualizer(model, dataset.cat_features)
        file_path = f"{folder}/{name}"
        visualizer.visualize(file_path=file_path, view=False)


if __name__ == "__main__":
    folder = "docs/charts/nn_diagrams"
    os.makedirs(folder, exist_ok=True)

    # # Example usage
    # dataset = ConflictingDataset()
    # model = dataset.generate_model()
    # visualizer = ModelVisualizer(model, dataset.cat_features)
    # dot = visualizer.visualize(file_path=f"{folder}/conflicting", view=False)

    save_all_datasets(folder=f"{folder}")

    from xaiunits.model.boolean_and import AND
    from xaiunits.model.boolean_not import NOT
    from xaiunits.model.boolean_or import OR

    model = AND()
    visualizer = ModelVisualizer(model, cat_features=[0, 1])
    dot = visualizer.visualize(file_path=f"{folder}/boolean_and", view=False)
    model = OR()
    visualizer = ModelVisualizer(model, cat_features=[0, 1])
    dot = visualizer.visualize(file_path=f"{folder}/boolean_or", view=False)
    model = NOT(1)
    visualizer = ModelVisualizer(model, cat_features=[0])
    dot = visualizer.visualize(file_path=f"{folder}/boolean_not", view=False)

    plot_legend(folder=f"{folder}")

    image_path = f"{folder}/boolean_not.png"
    larger_image_path = f"{folder}/boolean_and.png"
    output_path = f"{folder}/boolean_not_padded.png"

    # Open the larger image to get its size
    larger_image = Image.open(larger_image_path)
    larger_image_size = larger_image.size

    # Pad the smaller image
    pad_image_to_match_size(image_path, larger_image_size, output_path)
