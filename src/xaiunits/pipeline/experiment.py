import inspect
from collections.abc import Iterable

import torch
from torch.utils.data import Dataset
from xaiunits.metrics import wrap_metric
from typing import List, Tuple, Optional, Any, Callable, Dict, Union


class Experiment:
    """
    A class representing an experimental setup for evaluating explanation methods on specific dataset and
    neural network models.

    It should be ensured that the class corresponding to the dataset contains the generate_model() method
    in order for a model to be generated in the case that no model is defined for the experiment at
    initialization.

    Attributes:
        data (torch.utils.data.Dataset): The class of the dataset used for experiments.
        models (list | NoneType): List of neural network models to apply the explanation methods on.
        methods (list): List of explanation methods to apply and evaluate.
        metrics (list): List of evaluation metrics to compute.
        seeds (list): List of random seeds to use for the instantiation of the dataset.
        method_seeds (list): List of random seeds to use for explanation methods.
        data_params (dict): Additional parameters to be passed to the instantiation of the dataset.
    """

    def __init__(
        self,
        data: Any,
        models: Union[List[torch.nn.Module], torch.nn.Module, None],
        methods: Union[Any, List[Any]],
        metrics: Optional[Union[Any, List[Any]]] = None,
        seeds: int = 0,
        method_seeds: int = 0,
        data_params: Optional[Dict] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Initializes an Experiment instance.

        Args:
            data (Any): The dataset or data class to be used for experiments. Can be passed in as
                an instantiated object or as a subclass of torch.utils.data.Dataset.
            models (list | NoneType | torch.nn.Module): List of neural network models to apply the
                explanation methods on.
            methods (Any): List of explanation methods to apply and evaluate.
            metrics (Any | NoneType, optional): List of evaluation metrics to compute. Defaults to None.
            seeds (int | list): List of random seeds to use for the instantiation of the dataset.
                Defaults to 0.
            method_seeds (int | list): List of random seeds to use for explanation methods.
                Defaults to 0.
            data_params (dict | NoneType, optional): Additional parameters to be passed during the instantiation
                of the dataset. Defaults to None.
            name (str, optional): string representing name of the experiment.

        Raises:
            Exception: If input to any attribute initialization method is invalid.
        """
        data_params = data_params if data_params is not None else {}

        self.seeds = self._init_seeds(seeds)
        self.data = self._init_data(data)
        self.data_params = self._init_data_params(data_params)
        self.models = self._init_models(models)
        self.metrics = self._init_metrics(metrics)
        self.methods = self._init_methods(methods)
        self.method_seeds = self._init_method_seeds(method_seeds)
        self.exp_name = name

    def get_data(self, seed: int) -> Any:
        """
        Returns the dataset instance generated with the specified seed.

        Args:
            seed (int): the seed for instantiating the dataset.

        Returns:
            torch.utils.data.Dataset: The dataset instance with the specified seed.
        """
        return self.data(seed=seed, **self.data_params)

    def get_models(self, data_instance: Any) -> List[torch.nn.Module]:
        """
        Returns the list of neural networks to apply the explanation methods on.

        A default neural network compatible with the given dataset will be generated
        if the Experiment object has None as its models.

        Args:
            data_instance (torch.utils.data.Dataset): The dataset instance.

        Returns:
            list: List of neural networks to apply the explanation methods on.
        """
        if self.models is not None:
            return self.models
        else:
            return [data_instance.generate_model()]

    def get_methods(self, data_instance: Any) -> List[Any]:
        """
        Returns the list of explanation methods to apply and evaluate.

        Args:
            data_instance (torch.utils.data.Dataset): The dataset instance and
                a placeholder to keep input standardized.

        Returns:
            list: List of explanation methods to apply and evaluate.
        """
        return self.methods

    def get_metrics(self, data_instance: Any) -> List[Any]:
        """
        Returns the list of evaluation metrics to compute.

        Args:
            data_instance (torch.utils.data.Dataset): The dataset instance.

        Returns:
            list: List of evaluation metrics to compute.
        """
        metrics = []
        for metric in self.metrics:
            if metric is None:
                metrics.append(data_instance.default_metric)
            elif inspect.isclass(metric):
                metrics.append(metric)
            elif isinstance(metric, dict):
                if metric["metric_fns"].__name__ == "infidelity":
                    metric["perturb_func"] = data_instance.perturb_function()
                metrics.append(wrap_metric(**metric))
        return metrics

    def _init_seeds(self, seeds: int) -> List[int]:
        """
        Initializes the seeds attribute and transforms the input to the desired datatype.

        Args:
            seeds (int | list): Random seeds to use for data.

        Returns:
            list: List of random seeds to use for the instantiation of the dataset.

        Raises:
            Exception: If input to seeds initialization is not an integer or an Iterable of integers.
        """
        if type(seeds) == int:
            return [seeds]
        if isinstance(seeds, Iterable):
            if all(type(seed) is int for seed in seeds):
                return seeds
        raise Exception("Invalid input to data seeds.")

    def _init_data(self, data: Any) -> Any:
        """
        Initializes the data attribute.

        Args:
            data (type): The instantiated dataset or data class.

        Returns:
            torch.utils.data.Dataset | type: The dataset or data class.

        Raises:
            Exception: If input to data initialization is neither a torch.utils.data.Dataset instance
                or subclass of torch.utils.data.Dataset.
        """
        if isinstance(data, Dataset):
            return data
        try:
            if issubclass(data, Dataset):
                return data
        except TypeError:
            raise Exception("Invalid input to data class.")

        raise Exception("Invalid input to data class.")

    def _init_methods(self, methods: Any) -> Any:
        """
        Initializes the methods attribute.

        Args:
            methods (list | NoneType): List of explanation methods.

        Returns:
            list: List of explanation methods.

        Raises:
            Exception: If input to methods initialization is None.
        """
        if methods is None:
            raise Exception(
                "Invalid input to explanation methods. Needs to be a list of feature attribution methods."
            )
        else:
            return methods

    def _init_metrics(self, metrics: Any) -> Any:
        """
        Initializes the metrics attribute.

        Args:
            metrics (list | NoneType): List of evaluation metrics.

        Returns:
            list: List of evaluation metrics.

        Raises:
            Exception: If input to metrics initialization is None and the dataset
                does not provide a default metric.
        """
        if metrics is None:
            if hasattr(self.data, "default_metric"):
                return [None]
            else:
                raise Exception(
                    "Invalid input to evaluation metrics. Needs to be a list of evaluation metrics."
                )
        else:
            self._verify_metric(metrics)
            return metrics

    def _init_models(self, models: Any) -> Any:
        """
        Initializes the models attribute and transforms it to the desired datatype.

        Args:
            models (list | torch.nn.Module): Neural network models.

        Returns:
            list: List of neural network models.

        Raises:
            Exception: If input to models initialization is not an torch.nn.Module object or
                Iterable of torch.nn.Module objects.
        """
        if models is not None:
            if isinstance(models, torch.nn.Module):
                return [models]
            if isinstance(models, Iterable):
                if all(isinstance(model, torch.nn.Module) for model in models):
                    return models
            raise Exception("Invalid input to models.")

        else:
            return models

    def _init_data_params(self, data_params: Dict) -> Dict:
        """
        Initializes the data_params attribute.

        Args:
            data_params (dict): Additional parameters for the instantiation of the dataset.

        Returns:
            dict: Dictionary of additional data parameters.

        Raises:
            Exception: If input to data parameters initialization is not a dictionary.
        """
        if type(data_params) == dict:
            return data_params
        else:
            raise Exception("Invalid input to data parameters.")

    def _init_method_seeds(
        self, method_seeds: Union[int, Iterable[int]]
    ) -> Iterable[int]:
        """
        Initializes the method seeds attribute and transforms the input to the desired datatype.

        Args:
            method_seeds (int | list): Random seeds to use for applying the explanation
                methods.

        Returns:
            list: List of random seeds for applying the explanation methods.

        Raises:
            Exception: If input to method seeds initialization is not an integer nor an Iterable
                of integers.
        """
        if type(method_seeds) == int:
            return [method_seeds]
        if isinstance(method_seeds, Iterable):
            if all(type(seed) is int for seed in method_seeds):
                return method_seeds
        raise Exception("Invalid input to method seeds.")

    def _verify_metric(self, metrics: Any) -> None:
        """
        Verifies whether evaluation metrics have unique labels and each of them is in a valid datatype.

        Args:
            metrics (list): A list of evaluation metrics.

        Raises:
            Exception: If evaluation metric is defined in an invalid datatype or if evaluation metrics
                do not have unique labels.
        """
        metric_names = []
        for metric in metrics:
            if isinstance(metric, dict):
                metric_name = metric.get("name", metric["metric_fns"])
            elif isinstance(metric, type):
                metric_name = metric.__name__
            else:
                raise Exception(
                    "Contains evaluation metric defined in an invalid datatype."
                )
            metric_names.append(metric_name)

        if len(metric_names) != len(set(metric_names)):
            raise Exception("Evaluation metrics must have unique names.")
