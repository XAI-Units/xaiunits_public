import random
from typing import Tuple, List, Optional, Callable, Any, Iterable

import torch
from torch import nn
from sympy import symbols
from sympy.logic.boolalg import truth_table
from sympy.core.function import FunctionClass
from xaiunits.datagenerator.data_generation import BaseFeaturesDataset
from xaiunits.metrics import perturb_func_constructor


class BooleanDataset(BaseFeaturesDataset):
    """
    Generic synthetic dataset based on a propositional formula.

    The dataset corresponds to sampling rows from the truth table of the given propositional
    formula. If n_samples is no larger than the size of the truth table, then the generated
    dataset will always contain non-duplicate samples of the truth table. Otherwise,
    the dataset will still contain rows for the entire truth table but will also contain
    duplicates.

    If the input for atoms is None, the corresponding attribute is by default assigned
    as the atoms that are extracted from the given formula.

    Inherits from:
        BaseFeaturesDataset: The base class for creating continuous feature datasets.

    Attributes:
        formula (sympy.core.function.FunctionClass): A propositional formula for which the dataset
            is generated.
        atoms (tuple): The ordered collection of propositional atoms that were used within the
            propositional formula.
        seed (int): Seed for random number generators to ensure reproducibility.
        n_samples (int): Number of samples in the dataset.
    """

    def __init__(
        self,
        formula: FunctionClass,
        atoms: Optional[Iterable] = None,
        seed: int = 0,
        n_samples: int = 10,
    ):
        """
        Initializes a BooleanDataset object.

        Args:
            formula (sympy.core.function.FunctionClass): A propositional formula for dataset generation.
            atoms (Iterable, optional): Ordered collection of propositional atoms used in the formula.
                Defaults to None.
            seed (int): Seed for random number generation, ensuring reproducibility. Defaults to 0.
            n_samples (int): Number of samples to generate for the dataset. Defaults to 10.
        """
        self.atoms = tuple(formula.atoms()) if atoms is None else atoms
        self.formula = formula

        self.seed, self.n_features, self.n_samples = self._validate_inputs(
            seed, len(self.atoms), n_samples
        )
        random.seed(seed)
        torch.manual_seed(seed)
        self.samples, self.labels = self._initialize_samples_labels(n_samples)
        self.features = self.ground_truth_attribute = "samples"
        self.subset_data = ["samples"]
        self.subset_attribute = [
            "perturb_function",
            "default_metric",
            "generate_model",
            "name",
        ]
        self.cat_features = list(range(self.n_features))
        self.name = self.__class__.__name__

    def _initialize_samples_labels(
        self, n_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes the samples and labels of the dataset.

        Args:
            n_samples (int): number of samples/labels contained in the dataset.

        Returns:
            tuple[Tensor, Tensor]: Tuple containing the generated samples
                and corresponding labels of the dataset.
        """
        _truth_table = [tup for tup in truth_table(self.formula, self.atoms)]

        # sample random indices
        indices = random.sample(
            range(len(_truth_table)), min(len(_truth_table), n_samples)
        )
        if n_samples > len(_truth_table):
            indices += random.choices(
                range(len(_truth_table)), k=n_samples - len(_truth_table)
            )
        random.shuffle(indices)

        # replace booleans by 1.0 or -1.0
        samples = torch.Tensor([tup[0] for tup in _truth_table])
        samples = torch.where(samples == 1, 1.0, -1.0)
        labels = torch.Tensor([1.0 if tup[1] else -1.0 for tup in _truth_table])
        return samples[indices], labels[indices]

    def perturb_function(
        self,
        cat_resample_prob: float = 0.2,
        run_infidelity_decorator: bool = True,
        multipy_by_inputs: bool = False,
    ) -> Callable:
        """
        Generates perturb function to be used for XAI method evaluation. Applies gaussian noise
        for continuous features, and resampling for categorical features.

        Args:
            cat_resample_prob (float): Probability of resampling a categorical feature. Defaults to 0.2.
            run_infidelity_decorator (bool): Set to true if the returned fns is to be compatible with
                infidelity. Set flag to False for sensitivity. Defaults to True.
            multiply_by_inputs (bool): Parameters for decorator. Defaults to False.

        Returns:
            perturb_func (function): A perturbation function compatible with Captum.
        """
        return perturb_func_constructor(
            noise_scale=0.0,  # Only categorical
            cat_resample_prob=cat_resample_prob,
            cat_features=self.cat_features,
            replacements={
                cat_feature: [-1.0, 1.0] for cat_feature in self.cat_features
            },
            run_infidelity_decorator=run_infidelity_decorator,
            multipy_by_inputs=multipy_by_inputs,
        )

    def generate_model(self) -> torch.nn.Module:
        """
        Generates a neural network model using the given propositional formula and atoms.

        Returns:
            model.PropFormulaNN: A neural network model tailored to the dataset's propositional formula.
        """
        from xaiunits.model.boolean import PropFormulaNN

        return PropFormulaNN(self.formula, self.atoms)

    @property
    def default_metric(self) -> Callable:
        """
        The default metric for evaluating the performance of explanation methods applied
        to this dataset.

        For this dataset, the default metric is the infidelity metric with the
        default perturb function.

        Returns:
            type: A class that wraps around the default metric to be instantiated
                within the pipeline.
        """
        from captum.metrics import infidelity
        from xaiunits.metrics import wrap_metric

        return wrap_metric(infidelity, perturb_func=self.perturb_function())

    def __getitem__(self, idx: int, others: List[str] = []) -> Tuple[Any, ...]:
        """
        Retrieve a sample and its associated label by index.

        Args:
            idx (int): Index of the sample to retrieve.
            others (list): Additional items to retrieve. Defaults to [].

        Returns:
            tuple: Tuple containing the sample and its label.
        """
        return super().__getitem__(idx, others=others)


class BooleanAndDataset(BooleanDataset):
    def __init__(self, n_features: int = 2, n_samples: int = 10, seed: int = 0) -> None:
        assert n_features >= 2
        self.n_features = n_features
        atoms = symbols(",".join([f"{i}" for i in range(n_features)]))
        k = atoms[0] & atoms[1]
        for i in range(2, n_features):
            k = k & atoms[i]
        super().__init__(k, atoms, n_samples=n_samples, seed=seed)

        self.ground_truth = self.create_ground_truth()

        self.ground_truth_attribute = "ground_truth"
        self.create_baselines()

    def create_baselines(self) -> None:
        self.baseline = torch.ones_like(self.samples) * 1
        self.baseline[(self.samples == self.baseline).all(dim=1)] = (
            torch.ones_like(self.samples[0]) * -1
        )

    def __getitem__(
        self, idx: int, others: List[str] = ["baseline", "ground_truth_attribute"]
    ) -> Tuple[Any, ...]:
        return super().__getitem__(idx, others=others)

    def generate_model(self) -> torch.nn.Module:
        from xaiunits.model.boolean_and import BooleanAndNN

        return BooleanAndNN(self.n_features)

    def create_ground_truth(self) -> torch.Tensor:
        mask = (self.samples == 1).all(dim=1)
        ground_truth = (self.samples == -1.0).float() * (
            -2.0 / (self.samples == -1.0).float().sum(dim=1, keepdim=True)
        )
        ground_truth[mask.nonzero()] = torch.tensor(
            [2.0 / self.samples.shape[1]] * self.samples.shape[1]
        )

        return ground_truth

    @property
    def default_metric(self) -> Callable:
        from xaiunits.metrics import wrap_metric

        return wrap_metric(
            torch.nn.functional.mse_loss,
            out_processing=lambda x: torch.mean(x.flatten(1), dim=1),
        )


class BooleanOrDataset(BooleanDataset):
    def __init__(self, n_features: int = 2, n_samples: int = 10, seed: int = 0) -> None:
        assert n_features >= 2
        self.n_features = n_features
        atoms = symbols(",".join([f"{i}" for i in range(n_features)]))
        k = atoms[0] | atoms[1]
        for i in range(2, n_features):
            k = k | atoms[i]
        super().__init__(k, atoms, n_samples=n_samples, seed=seed)

        self.ground_truth = self.create_ground_truth()

        self.ground_truth_attribute = "ground_truth"
        self.create_baselines()

    def create_baselines(self) -> None:
        self.baseline = torch.ones_like(self.samples) * -1
        self.baseline[(self.samples == self.baseline).all(dim=1)] = torch.ones_like(
            self.samples[0]
        )

    def __getitem__(
        self, idx: int, others: List[str] = ["baseline", "ground_truth_attribute"]
    ) -> Tuple[Any, ...]:
        return super().__getitem__(idx, others=others)

    def generate_model(self) -> torch.nn.Module:
        from xaiunits.model.boolean_or import BooleanOrNN

        return BooleanOrNN(self.n_features)

    def create_ground_truth(self) -> torch.Tensor:
        mask = (self.samples == -1).all(dim=1)
        ground_truth = (self.samples == 1.0).float() * (
            2.0 / (self.samples == 1.0).float().sum(dim=1, keepdim=True)
        )
        ground_truth[mask.nonzero()] = torch.tensor(
            [-2.0 / self.samples.shape[1]] * self.samples.shape[1]
        )

        return ground_truth

    @property
    def default_metric(self) -> Callable:
        from xaiunits.metrics import wrap_metric

        return wrap_metric(
            torch.nn.functional.mse_loss,
            out_processing=lambda x: torch.mean(x.flatten(1), dim=1),
        )


if __name__ == "__main__":
    from sympy import symbols

    # x, y, z, a = symbols("x y z a")
    # k = (x & (y | ~z)) & (z | a)
    # print(type(type(k)))
    # data = BooleanDataset(k, n_samples=10)

    data = BooleanAndDataset(n_samples=10000, n_features=10)

    ground_truth = (data.samples == -1.0).float() * (
        2 / (data.samples == -1.0).float().sum(dim=1, keepdim=True)
    )
    print(ground_truth)
    print()
