import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from xaiunits.datagenerator import WeightedFeaturesDataset
from typing import Optional, Tuple, Any, List, Dict


class ShatteredGradientsDataset(WeightedFeaturesDataset):
    """
    A class intended to generate data and weights that exhibit shattered gradient phenomena.

    This class generates weights depending on the activation function and the discontinuity ratios.
    The discontinuity ratio is a set of real numbers (one per feature), so small perturbations
    around this discontinuity ratio significantly impact the model's explanation.

    Inherits from:
        WeightedFeaturesDataset: Class extending BaseFeaturesDataset with support for weighted features

    Attributes:
        weights (Tensor): Weights applied to each feature.
        weight_range (tuple): The range (min, max) within which random weights are generated.
        weighted_samples (Tensor): The samples after applying weights.
    """

    def __init__(
        self,
        seed: int = 0,
        n_features: int = 5,
        n_samples: int = 100,
        discontinuity_ratios: Optional[List] = None,
        bias: float = 0.5,
        act_fun: str = "Relu",
        two_distributions_flag: bool = False,
        proportion: float = 0.2,
        classification: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a ShatteredGradientsDataset object.

        Args:
            seed (int): Seed for reproducibility. Defaults to 0.
            n_features (int): Number of features. Defaults to 5.
            n_samples (int): Number of samples. Defaults to 100.
            discontinuity_ratios (list, optional): Ratios indicating feature discontinuity.
                If None, ratios are generated randomly. Defaults to None. Example: (1, -3, 4, 2, -2)
            bias (float): Bias value. Defaults to 0.5.
            act_fun (str): Activation function ("Relu", "Gelu", or "Sigmoid"). Defaults to "Relu".
            two_distributions_flag (bool): Flag for using two distributions. Defaults to False.
            proportion (float): Proportion of samples for narrow distribution when using two distributions.
                Defaults to 0.2.
            classification (bool): Flag for classification. Defaults to False.
            **kwargs: Arbitrary keyword arguments passed to the base class constructor, including:

                - sample_std_dev_narrow (float): Standard deviation for sample creation noise in
                  narrow distribution. Defaults to 0.05.
                - sample_std_dev_wide (float): Standard deviation for sample creation noise in
                  wide distribution. Defaults to 10.
                - weight_scale (float): Scalar value to multiply all generated weights with.
                - label_std_dev (float): Noise standard deviation to generate labels. Defaults to 0.
        """
        torch.manual_seed(seed)

        if two_distributions_flag:
            self._initialize_with_narrow_wide_distributions(
                seed,
                n_features,
                n_samples,
                discontinuity_ratios,
                bias,
                act_fun,
                proportion,
                classification,
                kwargs,
            )
        else:
            self._initialize_with_narrow_distribution(
                seed,
                n_features,
                n_samples,
                discontinuity_ratios,
                bias,
                act_fun,
                classification,
                kwargs,
            )

    def _initialize_with_narrow_wide_distributions(
        self,
        seed: int,
        n_features: int,
        n_samples: int,
        discontinuity_ratios: List,
        bias: float,
        act_fun: str,
        proportion: float,
        classification: bool,
        kwargs: Optional[Dict],
    ) -> None:
        """
        Initializes the dataset with narrow and wide distributions.

        This method sets up the dataset with narrow and wide distributions. It generates
        a dataset with the first portion of data belonging to the narrow distribution
        dependent on sample_std_dev_narrow. Similarly, the second portion of the dataset will
        belong to the wider distribution, depending on sample_std_dev_wide.

        It also initializes the weights dependent on discontinuity ratios and weight_scale.

        Args:
            seed (int): Seed for random number generation to ensure reproducibility.
            n_features (int): Number of features in the dataset.
            n_samples (int): Number of samples in the dataset.
            discontinuity_ratios (list): List of discontinuity ratios for each feature.
            bias (float): Bias value to adjust the weight scale.
            act_fun (str): Activation function name ('Relu', 'Gelu', or 'Sigmoid').
            proportion (float): Proportion of narrow samples to wide samples.
            classification (bool): Indicates if the dataset is for classification (True) or regression (False).
            **kwargs: Arbitrary keyword arguments passed to the base class constructor, including:

                - sample_std_dev_narrow (float): Standard deviation for sample creation noise in
                  narrow distribution. Defaults to 0.05.
                - sample_std_dev_wide (float): Standard deviation for sample creation noise in
                  wide distribution. Defaults to 10.
                - weight_scale (float): Scalar value to multiply all generated weights with.
                - label_std_dev (float): Noise standard deviation to generate labels. Defaults to 0.
        """
        super().__init__(
            seed=seed,
            n_features=n_features,
            n_samples=n_samples,
            **kwargs,
        )

        self.discontinuity_ratios = self._initialize_discontinuity_ratios(
            discontinuity_ratios, n_features
        )
        self.bias = bias
        distribution_narrow, _ = self._get_default_distribution_narrow(
            n_features, kwargs
        )
        distribution_wide, kwargs = self._get_default_distribution_wide(
            n_features, kwargs
        )
        kwargs = self._get_weight_scale(kwargs, act_fun)
        weights = self._generate_default_weights(
            n_features, weight_scale=kwargs["weight_scale"], act_fun=act_fun
        )
        # self.seed, self.n_features, self.n_samples= self._validate_inputs(seed, n_features, n_samples)
        # self.sample_std_dev, self.label_std_dev = self._init_noise_parameters(
        #     kwargs
        # )
        self.samples, self.distribution = self._initialize_samples_narrow_wide(
            n_samples, proportion, distribution_narrow, distribution_wide
        )
        self.cat_features = []
        self.weights, self.weight_range = weights, None
        self.weighted_samples = self.samples * self.weights
        self.label_noise = torch.randn(n_samples) * self.label_std_dev
        self.features = "samples"
        self.ground_truth_attribute = None  # not returned
        self.subset_data = ["samples", "weighted_samples"]
        # self.subset_attribute = ["weights"] # Defaults to Base
        self.act_fun = self._default_activation_function(act_fun, classification)
        if classification == True:
            self.labels = (
                self.act_fun(self.weighted_samples.sum(dim=1)) + self.label_noise
            ) > 0.5
        else:
            self.labels = (
                self.act_fun(self.weighted_samples.sum(dim=1)) + self.label_noise
            )

    def _initialize_with_narrow_distribution(
        self,
        seed: int,
        n_features: int,
        n_samples: int,
        discontinuity_ratios: List,
        bias: float,
        act_fun: str,
        classification: bool,
        kwargs: Optional[Dict],
    ):
        """
        Initializes the dataset with just a narrow distribution.

        It generates a dataset with the first portion of data belonging
        to the narrow distribution dependent on sample_std_dev_narrow.

        It also initializes the weights dependent on discontinuity ratios and weight_scale.

        Args:
            seed (int): Seed for random number generation to ensure reproducibility.
            n_features (int): Number of features in the dataset.
            n_samples (int): Number of samples in the dataset.
            discontinuity_ratios (list): List of discontinuity ratios for each feature.
            bias (float): Bias value to adjust the weight scale.
            act_fun (str): Activation function name ('Relu', 'Gelu', or 'Sigmoid').
            proportion (float): Proportion of narrow samples to wide samples.
            classification (bool): Indicates if the dataset is for classification (True) or regression (False).
            **kwargs: Arbitrary keyword arguments passed to the base class constructor, including:

                - sample_std_dev_narrow (float): Standard deviation for sample creation noise in
                  narrow distribution. Defaults to 0.05.
                - weight_scale (float): Scalar value to multiply all generated weights with.
                - label_std_dev (float): Noise standard deviation to generate labels. Defaults to 0.
        """
        self.discontinuity_ratios = self._initialize_discontinuity_ratios(
            discontinuity_ratios, n_features
        )
        self.bias = bias
        distribution, kwargs = self._get_default_distribution_narrow(n_features, kwargs)
        kwargs = self._get_weight_scale(kwargs, act_fun)
        weights = self._generate_default_weights(
            n_features, weight_scale=kwargs["weight_scale"], act_fun=act_fun
        )

        super().__init__(
            seed=seed,
            n_features=n_features,
            n_samples=n_samples,
            distribution=distribution,
            weights=weights,
            weight_range=None,
            **kwargs,
        )
        self.act_fun = self._default_activation_function(act_fun, classification)
        if classification == True:
            self.labels = (
                self.act_fun(self.weighted_samples.sum(dim=1)) + self.label_noise
            ) > 0.5
        else:
            self.labels = (
                self.act_fun(self.weighted_samples.sum(dim=1)) + self.label_noise
            )

    def _initialize_samples_narrow_wide(
        self,
        n_samples: int,
        proportion: float,
        distribution_narrow: torch.distributions.Distribution,
        distribution_wide: torch.distributions.Distribution,
    ) -> Tuple[torch.Tensor, torch.distributions.Distribution]:
        """
        Initializes synthetic samples with narrow and wide distributions.

        Args:
            n_samples (int): Total number of samples to generate.
            proportion (float): Proportion of samples that should belong to the narrow distribution.
                It should be between 0 and 1, where 0 indicates no narrow samples, and 1 indicates all samples are narrow.
            distribution_narrow (torch.distributions.Distribution): Narrow distribution object.
            distribution_wide (torch.distributions.Distribution): Wide distribution object.

        Returns:
            tuple: A tuple containing the generated samples and the distribution used.
        """
        n_samples_narrow = int(n_samples * proportion)
        n_samples_wide = n_samples - n_samples_narrow
        samples_narrow, _ = self._init_samples(n_samples_narrow, distribution_narrow)
        samples_wide, distribution = self._init_samples(
            n_samples_wide, distribution_wide
        )
        samples = torch.cat((samples_narrow, samples_wide), dim=0)
        return samples, distribution

    def _initialize_discontinuity_ratios(
        self, discontinuity_ratios: Optional[List], n_features: int
    ) -> List[torch.Tensor]:
        """
        Initialize discontinuity ratios for each feature in the dataset.

        If `discontinuity_ratios` is None, this method generates initial discontinuity ratios for each feature
        based on the specified `n_features`.

        Args:
            discontinuity_ratios (list | NoneType): List of discontinuity ratios for each feature.
                If None, new discontinuity ratios will be generated.
            n_features (int): Number of features in the dataset.

        Returns:
            list: List of discontinuity ratios for each feature.

        Raises:
            AssertionError: If there are no positive or negative ratios, if `discontinuity_ratios`
                is not a list, or if the length of `discontinuity_ratios` does not match `n_features`.
        """
        if discontinuity_ratios is None:
            discontinuity_ratios = [
                torch.randint(-5, 5, ()).item() for _ in range(n_features)
            ]
            has_positive = any(ratio > 0 for ratio in discontinuity_ratios)
            has_negative = any(ratio < 0 for ratio in discontinuity_ratios)
            if not has_positive:
                discontinuity_ratios[0] = torch.randint(1, 6, ()).item()

            if not has_negative:
                index_to_change = (
                    1 if not has_positive and len(discontinuity_ratios) > 1 else 0
                )
                discontinuity_ratios[index_to_change] = -torch.randint(1, 6, ()).item()

        assert any(
            ratio > 0 for ratio in discontinuity_ratios
        ), "There must be at least one positive ratio"
        assert any(
            ratio < 0 for ratio in discontinuity_ratios
        ), "There must be at least one negative ratio"
        assert isinstance(discontinuity_ratios, list), "Ratios must be a list obj"
        assert (
            len(discontinuity_ratios) == n_features
        ), "Discontinuity must be specified for each feature"
        return discontinuity_ratios

    def _get_default_distribution_narrow(
        self, n_features: int, kwargs: Optional[Dict]
    ) -> Tuple[torch.distributions.Distribution, Dict]:
        """
        Returns the default narrow distribution for the dataset.

        This method sets the default narrow distribution based on the provided `kwargs` or defaults.
        The sample_std_dev_narrow is used to determine the covariance matrix of the distribution.

        Args:
            n_features (int): Number of features in the dataset.
            kwargs (dict): Additional keyword arguments for configuration:

                - sample_std_dev_narrow (float): Used to determine the covariance
                  matrix of the distribution.

        Returns:
            tuple: A tuple containing the default narrow distribution and the modified kwargs.
        """
        kwargs["sample_std_dev_narrow"] = kwargs.get(
            "sample_std_dev_narrow", kwargs.get("sample_std_dev", 0.05)
        )

        covariance_matrix = (kwargs["sample_std_dev_narrow"] ** 2) * torch.eye(
            n_features
        )
        distribution = MultivariateNormal(torch.zeros(n_features), covariance_matrix)

        return distribution, kwargs

    def _get_default_distribution_wide(
        self, n_features: int, kwargs: Optional[Dict]
    ) -> Tuple[torch.distributions.Distribution, Dict]:
        """
        Returns the default wide distribution for the dataset.

        This method sets up the default wide distribution based on the provided `kwargs` or defaults.
        The sample_std_dev_wide is used to determine the covariance matrix of the distribution.

        Args:
            n_features (int): Number of features in the dataset.
            kwargs (dict): Additional keyword arguments for configuration:

                - sample_std_dev_wide (float): Used to determine the covariance
                  matrix of the distribution.

        Returns:
            tuple: A tuple containing the default wide distribution and the modified kwargs.
        """
        kwargs["sample_std_dev_wide"] = kwargs.get("sample_std_dev_wide", 10)

        covariance_matrix = (kwargs["sample_std_dev_wide"] ** 2) * torch.eye(n_features)
        distribution = MultivariateNormal(torch.zeros(n_features), covariance_matrix)

        return distribution, kwargs

    def _default_activation_function(
        self, act_fun: str, classification: bool
    ) -> torch.nn.Module:
        """
        Returns the default activation function based on the provided function name and task type.

        Args:
            act_fun (str or nn.Module): Name or instance of the activation function ('Relu', 'Gelu', 'Sigmoid'),
                or a custom activation function instance.
            classification (bool): Indicates if the dataset is for classification (True) or regression (False).

        Returns:
            nn.Module: The default activation function is based on the specified name, instance, and task type.

        Raises:
            KeyError: If the provided activation function is not one of 'Relu', 'Gelu', or 'Sigmoid',
                and it does not match the type of a custom activation function already defined in the mapping.
        """
        mapping = {"Relu": nn.ReLU(), "Gelu": nn.GELU(), "Sigmoid": nn.Sigmoid()}
        if classification:
            act_fun = nn.Sigmoid()
        elif mapping.get(act_fun, False):
            act_fun = mapping[act_fun]
        elif [x for x in mapping.values() if type(x) == type(act_fun)]:
            act_fun = act_fun
        else:
            raise KeyError(
                "Activation Function can only be of Type Relu, Gelu or Sigmoid."
            )

        return act_fun

    def _get_weight_scale(self, kwargs: Optional[Dict], act_fun: str) -> Dict:
        """
        Adjust the weight scaling factor based on the activation function used.

        This method calculates and updates the weight scaling factor in the kwargs dictionary
        based on the provided activation function. A different default weight scale
        is applied for' Sigmoid' activation than other activation functions.

        Args:
            kwargs (dict): Additional keyword arguments, potentially including 'weight_scale'.
                If the user does not specify weight_scale, Default is implemented.
            act_fun (str): Name of the activation function ('Relu', 'Gelu', or 'Sigmoid').

        Returns:
            dict: Updated kwargs with the 'weight_scale' value adjusted according to the activation function.

        Raises:
            KeyError: If the activation function is not one of 'Relu', 'Gelu', or 'Sigmoid'.
        """
        if act_fun == "Sigmoid":
            kwargs["weight_scale"] = kwargs.get(
                "weight_scale", int((1 / kwargs["sample_std_dev_narrow"]) * 5000)
            )
        else:
            kwargs["weight_scale"] = kwargs.get(
                "weight_scale", int((1 / kwargs["sample_std_dev_narrow"]) * 500)
            )
        return kwargs

    def _generate_default_weights(
        self, n_features: int, weight_scale: float, act_fun: str
    ) -> torch.Tensor:
        """
        Generate default weights based on discontinuity ratios, bias, and activation function.

        Args:
            n_features (int): Number of features in the dataset.
            weight_scale (float): Scaling factor for weight initialization.
            act_fun (str): Name of the activation function ('Relu', 'Gelu', or 'Sigmoid').

        Returns:
            torch.Tensor: Default weights for each feature, adjusted based on discontinuity ratios, bias, and activation function.

        Raises:
            ZeroDivisionError: If the sum of positive or negative ratios is zero, indicating a configuration issue.
        """
        ratio_sign = [x > 0 for x in self.discontinuity_ratios]
        assert any(ratio_sign), "There must be at least one positive ratio"
        assert any(
            [(1 - value) for value in ratio_sign]
        ), "There must be at least one negative ratio"

        bias_sign = self.bias >= 0
        positive_magnitudes = sum([x for x in self.discontinuity_ratios if x >= 0])
        negative_magnitudes = sum([x for x in self.discontinuity_ratios if x < 0])

        if act_fun == "Sigmoid":
            negative_magnitudes += 2

        if bias_sign:
            positive_magnitudes += self.bias / weight_scale
        else:
            negative_magnitudes += self.bias / weight_scale

        if bias_sign:
            if negative_magnitudes != 0:
                beta = -positive_magnitudes / negative_magnitudes
            else:
                raise ZeroDivisionError(
                    "Change your negative ratios so that they do not sum up to zero"
                )
        else:
            if positive_magnitudes != 0:
                beta = -negative_magnitudes / positive_magnitudes
            else:
                raise ZeroDivisionError(
                    "Change your positive ratios so that they do not sum up to zero"
                )

        weights = [weight_scale] * n_features
        for i, two_distributions_flag in enumerate(ratio_sign):
            if (two_distributions_flag and not bias_sign) or (
                not two_distributions_flag and bias_sign
            ):
                weights[i] *= beta
        return torch.tensor(weights)

    def generate_model(self) -> torch.nn.Module:
        """
        Generate a model using the Shattered Gradients Neural Network architecture.

        Returns:
            model.ShatteredGradientsNN: An instance of the ShatteredGradientsNN model.
        """
        from xaiunits.model.shattered_gradients import ShatteredGradientsNN

        self.model = ShatteredGradientsNN(self.weights, act_fun=self.act_fun.__class__)
        return self.model

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

    @property
    def default_metric(self) -> None:
        """
        The default metric for evaluating the performance of explanation methods applied
        to this dataset.

        For this dataset, the default metric is the max sensitivity metric.

        Returns:
            type: A class that wraps around the default metric to be instantiated
                within the pipeline.
        """
        from captum.metrics import sensitivity_max
        from xaiunits.metrics import wrap_metric

        return wrap_metric(sensitivity_max)


if __name__ == "__main__":
    data = ShatteredGradientsDataset(n_samples=100)
    model = data.generate_model().float()
    inputs = data[:10][0]
    outputs = data[:10][1]
    print(data.weights)
    print(inputs)
    print(outputs)
    print(nn.functional.relu((inputs * data.weights).sum(dim=1)))
    print(model(inputs).squeeze())
