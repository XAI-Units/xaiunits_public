from unittest.mock import MagicMock

import pytest
import torch
from captum.attr import LRP, DeepLift, InputXGradient, KernelShap, ShapleyValueSampling
from captum.metrics import infidelity, sensitivity_max
from xaiunits.datagenerator import (
    PertinentNegativesDataset,
    WeightedFeaturesDataset,
)
from xaiunits.methods import wrap_method
from xaiunits.metrics import perturb_standard_normal, wrap_metric
from xaiunits.model import ContinuousFeaturesNN
from xaiunits.pipeline import (
    BasePipeline,
    Experiment,
    ExperimentPipeline,
    Pipeline,
    Results,
)


def equal_batch_results(batch1, batch2):
    assert batch1["batch_id"] == batch2["batch_id"]
    assert torch.all(batch1["batch_row_id"] == batch1["batch_row_id"])
    assert batch1["model"] == batch2["model"]
    assert batch1["method_seed"] == batch2["method_seed"]
    assert batch1["method"] == batch2["method"]
    assert batch1["data"] == batch2["data"]
    assert batch1["data_seed"] == batch2["data_seed"]
    assert type(batch1["attr_time"]) == type(batch2["attr_time"])
    assert torch.all(batch1["value"] == batch2["value"])
    return True


@pytest.fixture
def same_data_models():
    pass


@pytest.fixture
def data():
    return WeightedFeaturesDataset(n_features=5, n_samples=50)


@pytest.fixture
def random_input():
    return torch.randn((50, 5))


@pytest.fixture
def batch_results():
    return {
        "model": "ContinuousFeaturesNN",
        "method_seed": 1,
        "method": "shap",
        "data": "name",
        "data_seed": 0,
        "batch_id": 0,
        "batch_row_id": torch.arange(0, 50, 1),
    }


@pytest.fixture
def wrapped_method():
    return wrap_method(DeepLift)


@pytest.fixture
def model():
    return ContinuousFeaturesNN(5, torch.randn(5))


@pytest.fixture
def metrics():
    return [
        wrap_metric(
            torch.nn.functional.mse_loss,
            out_processing=lambda x: torch.mean(x.flatten(1), dim=1),
        )
    ]


@pytest.fixture
def pipeline():
    model = ContinuousFeaturesNN(5, torch.randn(5))
    data = WeightedFeaturesDataset(n_features=5, n_samples=50)
    method = wrap_method(DeepLift)
    metric = wrap_metric(
        torch.nn.functional.mse_loss,
        out_processing=lambda x: torch.mean(x.flatten(1), dim=1),
    )
    metric2 = wrap_metric(sensitivity_max)
    models = [model, model]
    methods = [method, method]
    metrics = [metric, metric2]
    datas = [data, data]
    method_seeds = [1, 2]
    return Pipeline(models, datas, methods, metrics, method_seeds)


class TestBasePipeline:
    def test_init(self):
        """Tests the __init__ method."""
        pipeline = BasePipeline(batch_size=10)
        assert pipeline.batch_size == 10
        assert pipeline.default_target is None
        assert pipeline.n_examples is None
        assert isinstance(pipeline.results, Results)

    def test_init_attr_iterable(self):
        """Tests the _init_attr method given Iterable input."""
        input = (1, 2)
        pipeline = BasePipeline()
        assert pipeline._init_attr(input) == [1, 2]

    def test_init_attr_str(self):
        """Tests the _init_attr method given string input."""
        input = "12"
        pipeline = BasePipeline()
        assert pipeline._init_attr(input) == [input]

    def test_init_attr_neural_network(self):
        """Tests the _init_attr method given torch.nn.Module input."""
        input = ContinuousFeaturesNN(1, torch.Tensor([1]))
        pipeline = BasePipeline()
        assert pipeline._init_attr(input) == [input]

    def test_single_explanation_attribute_type(
        self, data, model, wrapped_method, metrics
    ):
        """
        Tests for the type of the results object that is mutated
        by the _single_explanation_attribute method.
        """
        pipeline = BasePipeline()
        pipeline._single_explanation_attribute(
            data, model, wrapped_method, metrics, method_seed=0
        )
        assert len(pipeline.results.raw_data) == 1
        batch_result = pipeline.results.raw_data.pop()
        assert batch_result["batch_id"] == 0
        assert torch.all(batch_result["batch_row_id"] == torch.arange(0, 50, 1))
        assert batch_result["model"] == "ContinuousFeaturesNN"
        assert batch_result["method_seed"] == 0
        assert batch_result["method"] == "wrapper_DeepLift"
        assert batch_result["data"] == "WeightedFeaturesDataset"
        assert batch_result["data_seed"] == 0
        assert batch_result["metric"] == "mse_loss"
        assert type(batch_result["attr_time"]) == float
        assert type(batch_result["value"]) == torch.Tensor

    def test_single_explanation_attribute_correctness(
        self, data, model, wrapped_method, metrics
    ):
        """
        Tests for the correctness of the application of
        evaluation metric by the _single_explanation_attribute
        method.
        """
        pipeline = BasePipeline()
        pipeline._single_explanation_attribute(
            data, model, wrapped_method, metrics, method_seed=0
        )
        assert len(pipeline.results.raw_data) == 1
        batch_result = pipeline.results.raw_data.pop()
        metric_score = batch_result["value"]
        feature_inputs, y_labels, context = pipeline.unpack_batch(data[:])
        target = pipeline.set_default_target(model, feature_inputs, y_labels)
        metric = metrics[0]()
        method_instance = wrapped_method(model)
        attribute = pipeline._apply_attribution_method(
            feature_inputs, method_instance, {"temp": "temp"}
        )
        expected_score = metric(
            feature_inputs, y_labels, target, context, attribute, wrapped_method, model
        )
        assert torch.all(metric_score == expected_score)

    def test_apply_attribution_method(
        self, model, wrapped_method, random_input, batch_results
    ):
        """Tests the _apply_attribution_method for correctness."""
        pipeline = BasePipeline()
        method_instance = wrapped_method(model)
        expected_output = method_instance.attribute(random_input)
        output = pipeline._apply_attribution_method(
            random_input, method_instance, batch_results
        )
        assert torch.all(output == expected_output)
        assert "attr_time" in batch_results.keys()
        assert type(batch_results["attr_time"]) == float

    def test_store_top_n(self):
        """Tests the _store_top_n method by running a single example and checking it is stored"""
        xmethods = [InputXGradient]
        data = WeightedFeaturesDataset(n_samples=1)
        model = data.generate_model()

        metrics = [
            wrap_metric(
                torch.nn.functional.mse_loss,
                out_processing=lambda x: torch.mean(x.flatten(1), dim=1),
            )
        ]
        p = Pipeline(
            model,
            data,
            xmethods,
            metrics,
            method_seeds=[1, 0],
            default_target=0,
            n_examples=1,
        )
        p.explanation_attribute()

        max_example = p.results.examples["max"][
            ("InputXGradient", "ContinuousFeaturesNN", "mse_loss")
        ][0]
        min_example = p.results.examples["min"][
            ("InputXGradient", "ContinuousFeaturesNN", "mse_loss")
        ][0]

        assert torch.allclose(
            max_example.feature_inputs,
            data.samples[0],
        )
        assert torch.allclose(
            min_example.feature_inputs,
            data.samples[0],
        )
        # We also expect 0 score since this is both a ContinuousFeatures Dataset and Model
        assert max_example.score == 0.0
        assert min_example.score == 0.0

    def test_unpack_batch_single(self):
        """
        Tests the unpack_batch method when only a batch of data is passed.
        """
        batch = torch.randn((50, 5))
        pipeline = BasePipeline()
        output = pipeline.unpack_batch(batch)
        assert output[0] is batch
        assert output[1] is None
        assert output[2] is None

    def test_unpack_batch_double(self):
        """
        Tests the unpack_batch method when only a batch of data and labels
        are passed.
        """
        batch = (torch.randn((50, 5)), torch.randn((50, 5)))
        pipeline = BasePipeline()
        output = pipeline.unpack_batch(batch)
        assert output[0] is batch[0]
        assert output[1] is batch[1]
        assert output[2] is None

    def test_unpack_batch_triple(self):
        """
        Tests the unpack_batch method when a batch of data, labels, and
        are passed.
        """
        batch = (torch.randn((50, 5)), torch.randn((50, 5)), {"temp": "temp"})
        pipeline = BasePipeline()
        output = pipeline.unpack_batch(batch)
        assert torch.all(output[0] == batch[0])
        assert torch.all(output[1] == batch[1])
        assert output[2] == batch[2]

    def test_unpack_batch_error(self):
        """
        Tests the unpack_batch method when an invalid input is given.
        """
        batch = (
            torch.randn((50, 5)),
            torch.randn((50, 5)),
            {"temp": "temp"},
            torch.randn((50, 5)),
        )
        pipeline = BasePipeline()
        with pytest.raises(TypeError) as error:
            pipeline.unpack_batch(batch)
            assert error.value == "Dataset Not of Typical Format"

    def test_set_default_target_not_str(self, model, random_input):
        """
        Tests the set_default_target method when the default target
        is not a string.
        """
        y_labels = random_input
        pipeline = BasePipeline()
        default_target = pipeline.set_default_target(model, random_input, y_labels)
        assert default_target == pipeline.default_target

    def test_set_default_target_y_label(self, model, random_input):
        """
        Tests the set_default_target method when the default target
        is the y label.
        """
        y_labels = random_input
        pipeline = BasePipeline(default_target="y_labels")
        default_target = pipeline.set_default_target(model, random_input, y_labels)
        assert torch.all(default_target == y_labels)

    def test_set_default_target_predicted_class(self, model, random_input):
        """
        Tests the set_default_target method when the default target
        is the predicted class.
        """
        y_labels = random_input
        pipeline = BasePipeline(default_target="predicted_class")
        default_target = pipeline.set_default_target(model, random_input, y_labels)
        assert torch.all(default_target == torch.argmax(model(random_input), dim=-1))

    def test_set_default_target_error(self, model, random_input):
        """
        Tests the set_default_target method given an invalid input.
        """
        y_labels = random_input
        pipeline = BasePipeline(default_target="temp")
        with pytest.raises(ValueError) as error:
            pipeline.set_default_target(model, random_input, y_labels)
            assert (
                error.value
                == "default_target not recognised. String input should be 'y_labels' or 'predicted_class'"
            )

    def test_all_models_explanation_attributes(
        self, data, model, wrapped_method, metrics
    ):
        """
        Tests the _all_methods_explanation_attributes method with
        all wrapped methods.
        """
        models = [model, model]
        methods = [wrapped_method, wrapped_method]
        method_seeds = [0, 1]
        pipeline = BasePipeline()
        pipeline._all_models_explanation_attributes(
            data, models, methods, metrics, method_seeds
        )
        results = pipeline.results
        assert len(results.raw_data) == len(models) * len(methods) * len(method_seeds)
        single_pipeline = BasePipeline()
        single_pipeline._single_explanation_attribute(
            data, models[0], methods[0], metrics, method_seed=0
        )
        assert equal_batch_results(
            results.raw_data[0], single_pipeline.results.raw_data[0]
        )

    def test_all_models_explanation_attributes_unwrapped(
        self, data, model, wrapped_method, metrics
    ):
        """
        Tests the _all_methods_explanation_attributes method with
        unwrapped methods.
        """
        models = [model, model]
        methods = [DeepLift, wrapped_method]
        method_seeds = [0, 1, 2]
        pipeline = BasePipeline()
        pipeline._all_models_explanation_attributes(
            data, models, methods, metrics, method_seeds
        )
        results = pipeline.results
        assert len(results.raw_data) == len(models) * len(methods) * len(method_seeds)
        single_pipeline = BasePipeline()
        single_pipeline._single_explanation_attribute(
            data,
            models[0],
            wrap_method(methods[0], pre_fix=""),
            metrics,
            method_seed=0,
        )
        assert equal_batch_results(
            results.raw_data[0], single_pipeline.results.raw_data[0]
        )

    def test_all_models_explanation_attributes_model_eval(
        self, data, model, wrapped_method, metrics
    ):
        """
        Tests whether the _all_methods_explanation_attributes method
        changes the neural network models to evaluation mode.
        """
        models = [model, model]
        methods = [DeepLift, wrapped_method]
        method_seeds = [0, 1, 2]
        pipeline = BasePipeline()
        pipeline._all_models_explanation_attributes(
            data, models, methods, metrics, method_seeds
        )
        for m in models:
            assert not m.training


class TestPipeline:
    def test_init(self, model, data, wrapped_method, metrics):
        """Tests the __init__ method."""
        models = [model, model]
        datas = [data, data]
        methods = [wrapped_method, wrapped_method]
        metrics = [
            wrap_metric(
                torch.nn.functional.mse_loss,
                out_processing=lambda x: torch.mean(x.flatten(1), dim=1),
            )
        ]
        pipeline = Pipeline(models, datas, methods, metrics)
        assert pipeline.models == models
        assert pipeline.datas == datas
        assert pipeline.methods == methods
        assert pipeline.metrics == metrics
        assert pipeline.method_seed == [None]

    def test_init_default_metric(self, model, data, wrapped_method):
        """
        Tests the __init__ method when no evaluation metric is explictly given
        and only uses the default metric from the dataset.
        """
        models = [model, model]
        datas = [data, data]
        methods = [wrapped_method, wrapped_method]
        pipeline = Pipeline(models, datas, methods)
        assert pipeline.models == models
        assert pipeline.datas == datas
        assert pipeline.methods == methods
        assert pipeline.metrics().__dict__ == datas[0].default_metric().__dict__
        assert pipeline.method_seed == [None]

    def test_init_duplicate_metric(self, model, data, wrapped_method):
        models = [model, model]
        datas = [data, data]
        methods = [wrapped_method, wrapped_method]
        eval_metric = wrap_metric(
            torch.nn.functional.mse_loss,
            out_processing=lambda x: torch.mean(x.flatten(1), dim=1),
        )
        metrics = [eval_metric, eval_metric]
        with pytest.raises(Exception) as error:
            pipeline = Pipeline(models, datas, methods, metrics)
            assert str(error.value) == "Evaluation metrics must have unique names."

    def test_subclass(self):
        """Tests that the Pipeline class is a subclass of BasePipeline."""
        assert issubclass(Pipeline, BasePipeline)

    def test_explanation_attr(self, pipeline):
        """Tests for the correctness of the explanation_attribute method."""
        pipeline.explanation_attribute()
        results = pipeline.results.raw_data
        assert len(results) == len(pipeline.datas) * len(pipeline.models) * len(
            pipeline.methods
        ) * len(pipeline.method_seed) * len(pipeline.metrics)
        single_pipeline = BasePipeline()
        single_pipeline._all_models_explanation_attributes(
            pipeline.datas[0],
            pipeline.models[0:1],
            pipeline.methods[0:1],
            pipeline.metrics[0:1],
            pipeline.method_seed[0:1],
        )
        single_results = single_pipeline.results.raw_data
        assert equal_batch_results(results[0], single_results[0])

    def test_run_alias_for_explanation_attribute(self, monkeypatch, pipeline):
        mock_explanation_attribute = MagicMock(return_value="mocked output")
        monkeypatch.setattr(
            pipeline, "explanation_attribute", mock_explanation_attribute
        )
        device = torch.device("cpu")
        result = pipeline.run(device)
        mock_explanation_attribute.assert_called_once_with(device)
        assert result == "mocked output"

    def test_init_none_metric(self, pipeline):
        """
        Tests the _init_none_metric method when at least one dataset
        has a default metric.
        """

        class Temp:
            pass

        pipeline.datas = [Temp(), pipeline.datas[0]]
        metric = pipeline._init_none_metric()
        assert metric().__dict__ == pipeline.datas[-1].default_metric().__dict__

    def test_init_none_metric_error(self, pipeline):
        """Tests the _init_none_metric method when no dataset has a default metric."""

        class Temp:
            pass

        pipeline.datas = [Temp(), Temp()]
        with pytest.raises(Exception) as error:
            pipeline._init_none_metric()
            assert (
                str(error.value)
                == "No (default) metric provided to the dataset or the pipeline."
            )


@pytest.fixture
def experiment1():
    data_seeds1 = [0, 1]
    method_seeds1 = [11, 12]
    xmethods1 = [DeepLift, ShapleyValueSampling]
    metrics1 = [
        wrap_metric(
            torch.nn.functional.mse_loss,
            out_processing=lambda x: torch.mean(x.flatten(1), dim=1),
        ),
        wrap_metric(sensitivity_max),
    ]
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
def experiment2():
    data_seeds2 = [2]
    method_seeds2 = [10, 100]
    xmethods2 = [KernelShap, LRP]
    metrics2 = [
        wrap_metric(
            torch.nn.functional.mse_loss,
            out_processing=lambda x: torch.mean(x.flatten(1), dim=1),
        ),
        wrap_metric(infidelity, perturb_func=perturb_standard_normal),
    ]
    return Experiment(
        PertinentNegativesDataset(n_samples=50),
        None,
        xmethods2,
        metrics2,
        data_seeds2,
        method_seeds2,
        {"n_samples": 50},
    )


class TestExperimentPipeline:
    def test_init(self, experiment1, experiment2):
        """
        Tests the __init__ method when multiple Experiment objects
        are passed.
        """
        experiments = [experiment1, experiment2]
        pipeline = ExperimentPipeline(experiments)
        assert pipeline.experiments == experiments

    def test_init_single(self, experiment1):
        """
        Tests the __init__ method when a single Experiment object
        is passed.
        """
        experiments = experiment1
        pipeline = ExperimentPipeline(experiments)
        assert pipeline.experiments == [experiment1]

    def test_subclass(self):
        """
        Tests that the ExperimentPipeline class is a subclass of
        BasePipeline.
        """
        assert issubclass(ExperimentPipeline, BasePipeline)

    def test_explanation_attribute_data_class(self, experiment1):
        """
        Tests for the correctness of the explanation_attribute method
        when the dataset in the Experiment object is a subclass of
        torch.utils.data.Dataset.
        """
        pipeline = ExperimentPipeline(experiment1)
        pipeline.explanation_attribute()
        results = pipeline.results.raw_data[0]
        data_instance = experiment1.data(seed=0, **experiment1.data_params)
        single_pipeline = Pipeline(
            data_instance.generate_model(),
            data_instance,
            experiment1.methods,
            experiment1.metrics,
            experiment1.method_seeds[0],
        )
        single_pipeline.explanation_attribute()
        single_results = single_pipeline.results.raw_data[0]
        assert equal_batch_results(results, single_results)

    def test_explanation_attribute_data_instance(self, experiment2):
        """
        Tests for the correctness of the explanation_attribute method
        when the dataset in the Experiment object is an instance of
        torch.utils.data.Dataset.
        """
        pipeline = ExperimentPipeline(experiment2)
        pipeline.explanation_attribute()
        results = pipeline.results.raw_data[0]
        single_pipeline = Pipeline(
            experiment2.data.generate_model(),
            experiment2.data,
            experiment2.methods,
            experiment2.metrics,
            experiment2.method_seeds[0],
        )
        single_pipeline.explanation_attribute()
        single_results = single_pipeline.results.raw_data[0]
        assert equal_batch_results(results, single_results)

    def test_run_alias_for_explanation_attribute(self, monkeypatch, experiment1):
        pipeline = ExperimentPipeline(experiment1)
        mock_explanation_attribute = MagicMock(return_value="mocked output")
        monkeypatch.setattr(
            pipeline, "explanation_attribute", mock_explanation_attribute
        )
        device = torch.device("cpu")
        result = pipeline.run(device)
        mock_explanation_attribute.assert_called_once_with(device)
        assert result == "mocked output"


# this block is only executed when you run the file directly as `python this_script.py`
# but you should be running `pytest` or `pytest this_script.py`
if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
