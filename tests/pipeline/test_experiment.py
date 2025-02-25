import pytest

from xaiunits.pipeline import Experiment
from xaiunits.datagenerator import (
    WeightedFeaturesDataset,
)
from xaiunits.metrics import wrap_metric
from xaiunits.model import ContinuousFeaturesNN

from captum.attr import DeepLift, ShapleyValueSampling, Lime, LRP
from captum.metrics import sensitivity_max, infidelity
import torch


@pytest.fixture
def experiment():
    data_seeds1 = [0, 1]
    method_seeds1 = [11, 12]
    xmethods1 = [DeepLift, ShapleyValueSampling]
    metrics1 = [wrap_metric(sensitivity_max)]
    return Experiment(
        WeightedFeaturesDataset,
        None,
        xmethods1,
        metrics1,
        data_seeds1,
        method_seeds1,
        {"n_samples": 50, "n_features": 3},
    )


@pytest.fixture
def default_experiment():
    return Experiment(
        WeightedFeaturesDataset,
        None,
        [Lime, LRP],
    )


class TestExperiment:
    def test_init(self, experiment):
        """Tests the __init__ method of Experiment class."""
        assert experiment.data == WeightedFeaturesDataset
        assert experiment.seeds == [0, 1]
        assert experiment.method_seeds == [11, 12]
        assert experiment.models is None
        assert experiment.methods == [DeepLift, ShapleyValueSampling]
        assert type(experiment.metrics[0]) == type(wrap_metric(sensitivity_max))
        assert experiment.data_params == {"n_samples": 50, "n_features": 3}

    def test_init_default(self, default_experiment):
        """
        Tests the __init__ method of Experiment class using default inputs
        for named arguments.
        """
        assert default_experiment.data == WeightedFeaturesDataset
        assert default_experiment.seeds == [0]
        assert default_experiment.method_seeds == [0]
        assert default_experiment.models is None
        assert default_experiment.methods == [Lime, LRP]
        assert default_experiment.metrics == [None]
        assert default_experiment.data_params == {}

    def test_init_duplicate_metric_names(self):
        """
        Tests the __init__ method of Experiment class using inputs to metrics
        that contain identical names.
        """
        with pytest.raises(Exception) as error:
            Experiment(WeightedFeaturesDataset, None, [Lime, Lime])
            assert str(error.value) == "Evaluation metrics must have unique names."

    def test_get_data(self, experiment):
        """
        Tests the get_data method of Experiment class.
        """
        data = experiment.get_data(0)
        assert isinstance(experiment.get_data(0), WeightedFeaturesDataset)
        assert len(data.samples) == 50
        assert data.n_features == 3

    def test_get_models_none(self, experiment):
        """
        Tests the get_models method of Experiment class when no existing model
        has been inputed to the experiment at instantiation.
        """
        data = experiment.get_data(0)
        models = experiment.get_models(data)
        assert len(models) == 1
        assert isinstance(models[0], ContinuousFeaturesNN)

    def test_get_models(self, experiment):
        """
        Tests the get_models method of Experiment class with existing model
        being inputed to the experiment at instantiation.
        """
        model1 = ContinuousFeaturesNN(1, torch.Tensor([1]))
        model2 = ContinuousFeaturesNN(3, torch.Tensor([1, 2, 3]))
        experiment.models = [model1, model2]
        data = experiment.get_data(0)
        assert experiment.get_models(data) == [model1, model2]

    def test_get_methods(self, experiment):
        """
        Tests the get_methods method of Experiment class when the method attribute
        of the object is not None.
        """
        data = experiment.get_data(0)
        assert experiment.get_methods(data) == experiment.methods

    def test_get_metrics_none(self, experiment):
        """
        Tests the get_metrics method of Experiment class when the metrics in
        the experiment object is provided by the default metric from the dataset.
        """
        data = experiment.get_data(0)
        experiment.metrics = [None]
        metrics = experiment.get_metrics(data)
        assert len(metrics) == 1
        assert metrics[0]().__dict__ == data.default_metric().__dict__

    def test_get_metrics(self, experiment):
        """
        Tests the get_metrics method of Experiment class when the metrics in
        the experiment object do not require contain additional arguments.
        """
        data = experiment.get_data(0)
        metrics = experiment.get_metrics(data)
        assert metrics == experiment.metrics

    def test_get_metrics_with_argmuments(self, experiment):
        """
        Tests the get_metrics method of Experiment class when the metrics in
        the experiment object requires additional arguments.
        """
        data = experiment.get_data(0)
        argument_metric = {
            "metric_fns": torch.nn.functional.mse_loss,
            "out_processing": lambda x: x,
        }
        experiment.metrics.append(argument_metric)
        metrics = experiment.get_metrics(data)
        assert type(metrics.pop()) == type(wrap_metric(sensitivity_max))
        assert type(metrics.pop()) == type(wrap_metric(**argument_metric))

    def test_get_metrics_with_infidelity(self, experiment):
        """
        Tests the get_metrics method of Experiment class when the metrics in
        the experiment object includes infidelity
        """
        data = experiment.get_data(0)
        argument_metric = {"metric_fns": infidelity, "out_processing": lambda x: x}
        experiment.metrics.append(argument_metric)
        metrics = experiment.get_metrics(data)
        assert type(metrics.pop()) == type(wrap_metric(sensitivity_max))
        assert type(metrics.pop()) == type(wrap_metric(**argument_metric))

    def test_init_seeds_error(self, experiment):
        """Tests the _init_seeds method of Experiment class with invalid input."""
        with pytest.raises(Exception) as error:
            experiment._init_seeds(None)
        assert str(error.value) == "Invalid input to data seeds."

    def test_init_seeds_iterable(self, experiment):
        """Tests the _init_seeds method of Experiment class with Iterable input."""
        assert experiment._init_seeds([1]) == [1]

    def test_init_seeds_int(self, experiment):
        """Tests the _init_seeds method of Experiment class with integer input."""
        assert experiment._init_seeds(1) == [1]

    def test_init_methods_none(self, experiment):
        """Tests the _init_methods method of Experiment class with NoneType input."""
        with pytest.raises(Exception) as error:
            experiment._init_methods(None)
        assert (
            str(error.value)
            == "Invalid input to explanation methods. Needs to be a list of feature attribution methods."
        )

    def test_init_methods(self, experiment):
        """Tests the _init_methods method of Experiment class without error."""
        assert experiment._init_methods(1) == 1

    def test_init_metrics_none(self, experiment):
        """
        Tests the _init_metrics method of Experiment class with NoneType input to
        retrieve the default metric from dataset.
        """
        assert experiment._init_metrics(None) == [None]

    def test_init_metrics_none_error(self, experiment):
        """
        Tests the _init_metrics method of Experiment class with NoneType input and
        the dataset does not provide a default metric.
        """

        class Temp:
            pass

        experiment.data = Temp()
        with pytest.raises(Exception) as error:
            experiment._init_metrics(None)
        assert (
            str(error.value)
            == "Invalid input to evaluation metrics. Needs to be a list of evaluation metrics."
        )

    def test_init_metrics(self, experiment):
        """Tests the _init_methods method of Experiment class without error."""
        assert experiment._init_metrics([Lime, LRP]) == [Lime, LRP]

    def test_init_models_none(self, experiment):
        """Tests the _init_models method of Experiment class with NoneType input."""
        assert experiment._init_models(None) is None

    def test_init_models_single(self, experiment):
        """Tests the _init_models method of Experiment class with single input."""
        model = ContinuousFeaturesNN(1, torch.Tensor([1]))
        assert experiment._init_models(model) == [model]

    def test_init_models_iterable(self, experiment):
        """Tests the _init_models method of Experiment class with Iterable input."""
        model1 = ContinuousFeaturesNN(1, torch.Tensor([1]))
        model2 = ContinuousFeaturesNN(2, torch.Tensor([1, 2]))
        models = [model1, model2]
        assert experiment._init_models(models) == models

    def test_init_models_error(self, experiment):
        """Tests the _init_models method of Experiment class with invalid input."""
        with pytest.raises(Exception) as error:
            experiment._init_models(1)
        assert str(error.value) == "Invalid input to models."

    def test_init_data_params_invalid(self, experiment):
        """Tests the _init_data_params method of Experiment class with invalid input."""
        with pytest.raises(Exception) as error:
            experiment._init_data_params(None)
        assert str(error.value) == "Invalid input to data parameters."

    def test_init_data_params(self, experiment):
        """Tests the _init_data_params method of Experiment class with valid input."""
        assert experiment._init_data_params({1: 2}) == {1: 2}
        assert experiment._init_data_params({}) == {}

    def test_init_method_seeds_error(self, experiment):
        """Tests the _init_method_seeds method of Experiment class with invalid input."""
        with pytest.raises(Exception) as error:
            experiment._init_method_seeds(None)
        assert str(error.value) == "Invalid input to method seeds."

    def test_init_method_seeds_iterable(self, experiment):
        """Tests the _init_method_seeds method of Experiment class with Iterable input."""
        assert experiment._init_method_seeds([1]) == [1]

    def test_init_method_seeds_int(self, experiment):
        """Tests the _init_method_seeds method of Experiment class with integer input."""
        assert experiment._init_method_seeds(1) == [1]

    def test_verify_metric(self, default_experiment):
        """
        Tests the _verify_metric method of Experiment class with valid metrics and names.
        """
        experiments = [wrap_metric(sensitivity_max), {"metric_fns": infidelity}]
        default_experiment._verify_metric(experiments)

    def test_verify_metric_duplicate_names(self, default_experiment):
        """
        Tests if error is raised by _verify_metric method of Experiment class with
        metrics with duplicate names.
        """
        experiments = [
            wrap_metric(sensitivity_max, name="temp", pre_fix=""),
            {"metric_fns": sensitivity_max, "name": "temp"},
        ]
        with pytest.raises(Exception) as error:
            default_experiment._verify_metric(experiments)
            assert str(error.value) == "Evaluation metrics must have unique names."

    def test_verify_metric_invalid_metric(self, default_experiment):
        """
        Tests if error is raised by _verify_metric method of Experiment class with
        metrics defined using invalid datatype.
        """
        experiments = [
            wrap_metric(sensitivity_max, name="temp", pre_fix=""),
            sensitivity_max,
        ]
        with pytest.raises(Exception) as error:
            default_experiment._verify_metric(experiments)
            assert (
                str(error.value)
                == "Contains evaluation metric defined in an invalid datatype."
            )


# this block is only executed when you run the file directly as `python this_script.py`
# but you should be running `pytest` or `pytest this_script.py`
if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
