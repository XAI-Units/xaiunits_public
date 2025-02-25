import pandas as pd
import pytest
import torch
from captum.attr import InputXGradient
from xaiunits.datagenerator import WeightedFeaturesDataset
from xaiunits.metrics import wrap_metric
from xaiunits.pipeline import Example, Pipeline, Results


@pytest.fixture
def random_data_batch():
    data_batch = {
        "data": "d1",
        "batch_id": 1,
        "batch_row_id": torch.Tensor([0, 1, 2, 3]),
        "method": "m3",
        "model": "RandomNN",
        "seed": 10,
        "data_seed": 1,
        "attr_time": 8,
        "metric": "m1",
        "value": torch.arange(4),
    }
    return data_batch


@pytest.fixture
def random_other_data_batch():
    data_batch = {
        "data": "d2",
        "batch_id": 2,
        "batch_row_id": torch.Tensor([0, 1, 2, 3, 4]),
        "method": "m4",
        "model": "RandomNN",
        "seed": 20,
        "data_seed": 2,
        "attr_time": 5,
        "metric": "m2",
        "value": torch.arange(5),
    }
    return data_batch


class TestResults:
    def test_init(self):
        """Tests the __init__ method for the Results class."""
        results = Results()
        assert not results.raw_data
        assert not results.examples["max"]
        assert not results.examples["min"]

    def test_append(self, random_data_batch):
        """Tests the append method for the Results class."""
        batch = random_data_batch
        results = Results()
        results.append(batch)
        assert len(results.raw_data) == 1
        assert not results.examples["max"]
        assert not results.examples["min"]

    def test_data(self, random_data_batch, random_other_data_batch):
        """Tests the data property for the Results class."""
        batch_1, batch_2 = random_data_batch, random_other_data_batch
        results = Results()
        results.append(batch_1)
        results.append(batch_2)
        data = results.data
        expected_data = {
            "data": ["d1" for _ in range(4)] + ["d2" for _ in range(5)],
            "batch_id": [1, 1, 1, 1, 2, 2, 2, 2, 2],
            "batch_row_id": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "method": ["m3" for _ in range(4)] + ["m4" for _ in range(5)],
            "model": ["RandomNN" for _ in range(9)],
            "seed": [10, 10, 10, 10, 20, 20, 20, 20, 20],
            "data_seed": [1, 1, 1, 1, 2, 2, 2, 2, 2],
            "attr_time": [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "metric": ["m1" for _ in range(4)] + ["m2" for _ in range(5)],
            "value": [0, 1, 2, 3, 0, 1, 2, 3, 4],
        }
        pd.testing.assert_frame_equal(data, pd.DataFrame(expected_data))

    def test_process_data(self, random_data_batch, random_other_data_batch):
        """Tests the process_data method for the Results class."""
        batch_1, batch_2 = random_data_batch, random_other_data_batch
        results = Results()
        results.append(batch_1)
        results.append(batch_2)
        pd.testing.assert_frame_equal(results.data, results.process_data())

    def test_print_stats_columns(self, random_data_batch, random_other_data_batch):
        """
        Tests the columns from print_stats method are correct for
        the Results class.
        """
        batch_1, batch_2 = random_data_batch, random_other_data_batch
        results = Results()
        results.append(batch_1)
        results.append(batch_2)
        columns = results.print_stats().columns
        expected_columns = [
            ("mean", "m1"),
            ("std", "m1"),
            ("mean", "m2"),
            ("std", "m2"),
            ("mean", "attr_time"),
            ("std", "attr_time"),
        ]
        assert set(columns) == set(expected_columns)

    def test_print_stats_indices(self, random_data_batch, random_other_data_batch):
        """
        Tests the indices from print_stats method are correct for
        the Results class.
        """
        batch_1, batch_2 = random_data_batch, random_other_data_batch
        results = Results()
        results.append(batch_1)
        results.append(batch_2)
        indices = results.print_stats().index
        expected_indices = [
            ("d1", "RandomNN", "m3"),
            ("d2", "RandomNN", "m4"),
        ]
        assert set(indices) == set(expected_indices)

    def test_attr_time_summing_dataset(
        self, random_data_batch, random_other_data_batch
    ):
        """
        Tests the _attr_time_summing method when given time
        unit as dataset.
        """
        batch_1, batch_2 = random_data_batch, random_other_data_batch
        results = Results()
        results.append(batch_1)
        results.append(batch_2)
        data = ["d1" for _ in range(4)] + ["d2" for _ in range(5)] + ["d1", "d2"]
        batch_row_id = [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 4.0] + [0, 0]
        method = ["m3" for _ in range(4)] + ["m4" for _ in range(5)] + ["m3", "m4"]
        expected_df = {
            "data": data,
            "batch_id": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0],
            "batch_row_id": batch_row_id,
            "method": method,
            "model": ["RandomNN" for _ in range(11)],
            "seed": [10, 10, 10, 10, 20, 20, 20, 20, 20, 10, 20],
            "data_seed": [1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2],
            "metric": ["m1" for _ in range(4)]
            + ["m2" for _ in range(5)]
            + ["attr_time", "attr_time"],
            "value": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 5.0],
        }
        actual_df = results._attr_time_summing("dataset").dropna(ignore_index=True)
        pd.testing.assert_frame_equal(actual_df, pd.DataFrame(expected_df))

    def test_attr_time_summing_batch(self, random_data_batch, random_other_data_batch):
        """
        Tests the _attr_time_summing method when given time
        unit as batch.
        """
        batch_1, batch_2 = random_data_batch, random_other_data_batch
        results = Results()
        results.append(batch_1)
        results.append(batch_2)
        data = ["d1" for _ in range(4)] + ["d2" for _ in range(5)] + ["d1", "d2"]
        batch_row_id = [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 4.0] + [0, 0]
        method = ["m3" for _ in range(4)] + ["m4" for _ in range(5)] + ["m3", "m4"]
        expected_df = {
            "data": data,
            "batch_id": [1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2],
            "batch_row_id": batch_row_id,
            "method": method,
            "model": ["RandomNN" for _ in range(11)],
            "seed": [10, 10, 10, 10, 20, 20, 20, 20, 20, 10, 20],
            "data_seed": [1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2],
            "metric": ["m1" for _ in range(4)]
            + ["m2" for _ in range(5)]
            + ["attr_time", "attr_time"],
            "value": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 5.0],
        }
        actual_df = results._attr_time_summing("batch").dropna(ignore_index=True)
        pd.testing.assert_frame_equal(actual_df, pd.DataFrame(expected_df))

    def test_attr_time_summing_instance(
        self, random_data_batch, random_other_data_batch
    ):
        """
        Tests the _attr_time_summing method when given time
        unit as data instance.
        """
        batch_1, batch_2 = random_data_batch, random_other_data_batch
        results = Results()
        results.append(batch_1)
        results.append(batch_2)
        data = 2 * (["d1" for _ in range(4)] + ["d2" for _ in range(5)])
        method = 2 * (["m3" for _ in range(4)] + ["m4" for _ in range(5)])
        metric = ["m1" for _ in range(4)] + ["m2" for _ in range(5)]
        metric += ["attr_time" for _ in range(9)]
        value = [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        value += [float(2) for _ in range(4)] + [float(1) for _ in range(5)]
        expected_df = {
            "data": data,
            "batch_id": 2 * [1, 1, 1, 1, 2, 2, 2, 2, 2],
            "batch_row_id": 2 * [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "method": method,
            "model": 2 * ["RandomNN" for _ in range(9)],
            "seed": 2 * [10, 10, 10, 10, 20, 20, 20, 20, 20],
            "data_seed": 2 * [1, 1, 1, 1, 2, 2, 2, 2, 2],
            "metric": metric,
            "value": value,
        }
        actual_df = results._attr_time_summing("instance").dropna(ignore_index=True)
        pd.testing.assert_frame_equal(actual_df, pd.DataFrame(expected_df))

    def test_attr_time_summing_invalid_input(self):
        """Tests the _attr_time_summing method when given invalid input."""
        obj = Results()
        with pytest.raises(Exception) as e:
            obj._attr_time_summing("invalid_unit")
        assert (
            str(e.value)
            == "Invalid unit of attribution time inputed. Only 'dataset', 'batch', 'instance' are accepted."
        )


class TestExampleClass:
    def test_single_example_through_pipeline(self):
        """Run a single datapoint through the pipeline and check the example is that datapoint"""
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

    def test_heap_ordering_for_max_examples(self):
        low_score = Example(
            score=0.2,
            attribute=torch.tensor([0.5]),
            feature_inputs=torch.tensor([1.0]),
            y_labels=torch.tensor([1]),
            target=None,
            context=None,
            example_type="max",
        )
        high_score = Example(
            score=10.5,
            attribute=torch.tensor([0.5]),
            feature_inputs=torch.tensor([1.0]),
            y_labels=torch.tensor([1]),
            target=None,
            context=None,
            example_type="max",
        )
        assert high_score > low_score

    def test_heap_ordering_for_min_examples(self):
        high_score = Example(
            score=10.5,
            attribute=torch.tensor([0.5]),
            feature_inputs=torch.tensor([1.0]),
            y_labels=torch.tensor([1]),
            target=None,
            context=None,
            example_type="min",
        )
        low_score = Example(
            score=0.2,
            attribute=torch.tensor([0.5]),
            feature_inputs=torch.tensor([1.0]),
            y_labels=torch.tensor([1]),
            target=None,
            context=None,
            example_type="min",
        )
        assert low_score > high_score


# this block is only executed when you run the file directly as `python this_script.py`
# but you should be running `pytest` or `pytest this_script.py`
if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
