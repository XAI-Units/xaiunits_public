from xaiunits.metrics import *
import pytest
from captum.attr import DeepLift
from captum.metrics import infidelity
import torch.nn as nn


class TestDefaultAttributeInputGenerator:
    def valid_inputs(self, args={}):
        """Input for Pytest"""
        shape = (1, 5)
        metric = args.get("metric", lambda x: x)
        feature_inputs = args.get("feature_inputs", torch.rand(shape))
        y_labels = args.get("y_labels", torch.zeros((1, 1)))
        target = args.get("target", torch.zeros((1, 1)))
        context = args.get("context", {"ground_truth_attribute": torch.rand(shape)})
        attribute = args.get("attribute", torch.rand(shape))
        model = args.get("model", torch.nn.Linear(5, 1))
        method_instance = DeepLift(model)

        return (
            metric,
            feature_inputs,
            y_labels,
            target,
            context,
            attribute,
            method_instance,
            model,
        )

    # @pytest.fixture(scope="class")
    def invalid_metric(self):
        """Input for Pytest"""
        return self.valid_inputs({"metric": 2})

    def invalid_feature_inputs(self):
        """Input for Pytest"""
        return self.valid_inputs({"feature_inputs": 2})

    def invalid_y_labels(self):
        """Input for Pytest"""
        return self.valid_inputs({"y_labels": 2})

    def invalid_target(self):
        """Input for Pytest"""
        return self.valid_inputs({"target": ""})

    def invalid_context(self):
        """Input for Pytest"""
        return self.valid_inputs({"context": 2})

    def invalid_attribute(self):
        """Input for Pytest"""
        return self.valid_inputs({"attribute": 2})

    def invalid_method_instance(self):
        """Input for Pytest"""
        return self.valid_inputs({"method_instance": 2})

    def valid_input_captum_metric(self):
        """Input for Pytest"""
        return self.valid_inputs({"metric": infidelity})

    def valid_input_mse(self):
        """Input for Pytest"""
        return self.valid_inputs({"metric": torch.nn.functional.mse_loss})

    def test_input_type(self):
        """Tests for invalid input."""
        with pytest.raises(AssertionError):
            default_metric_input_generator(*self.invalid_metric())

        with pytest.raises(AssertionError):
            default_metric_input_generator(*self.invalid_feature_inputs())

        with pytest.raises(AssertionError):
            default_metric_input_generator(*self.invalid_y_labels())

        with pytest.raises(AssertionError):
            default_metric_input_generator(*self.invalid_target())

        with pytest.raises(AssertionError):
            default_metric_input_generator(*self.invalid_context())

        with pytest.raises(AssertionError):
            default_metric_input_generator(*self.invalid_attribute())

        with pytest.raises(TypeError):
            default_metric_input_generator(*self.invalid_method_instance())

    def test_output_captum(self):
        """
        Tests whether the function returns expected output when
        metric is from Captum.
        """
        inputs = self.valid_input_mse()

        assert type(default_metric_input_generator(*inputs)) == dict
        assert "target" in list(default_metric_input_generator(*inputs))
        assert "input" in list(default_metric_input_generator(*inputs))
        assert "reduction" in list(default_metric_input_generator(*inputs))

    def test_output_mse(self):
        """
        Tests whether the function returns expected output when
        metric is mse_loss.
        """
        inputs = self.valid_input_captum_metric()

        assert type(default_metric_input_generator(*inputs)) == dict
        assert "forward_func" in list(default_metric_input_generator(*inputs))
        assert "inputs" in list(default_metric_input_generator(*inputs))
        assert "attributions" in list(default_metric_input_generator(*inputs))
        assert "explanation_func" in list(default_metric_input_generator(*inputs))


@pytest.fixture(scope="class")
def fake_wrapper_input():
    """Tests for invalid input."""
    fake_metric = None

    def correct_metric():
        pass

    fake_in_processing = "Test"
    fake_out_processing = "Test"
    no_params_order_in_processing = lambda x: x

    return (
        fake_metric,
        correct_metric,
        fake_out_processing,
        fake_in_processing,
        no_params_order_in_processing,
    )


@pytest.fixture(scope="class")
def valid_metric():
    """Tests for invalid input."""

    def metric(inputs, dummy1=False, dummy2=False):
        if dummy1 and dummy2:
            return torch.ones_like(inputs) * 3
        elif dummy1:
            return torch.ones_like(inputs) * 2
        elif dummy2:
            return torch.ones_like(inputs)
        else:
            return torch.zeros_like(inputs)

    return metric


@pytest.fixture(scope="class")
def valid_feature_input():
    """Tests for invalid input."""
    feature_inputs = torch.rand((10, 5))
    return feature_inputs


@pytest.fixture(scope="class")
def valid_input_gen():
    """Tests for invalid input."""
    return lambda metric, feature_input, y_labels, target, context, attribute, method_instance, model: {
        "inputs": feature_input
    }


@pytest.fixture(scope="class")
def valid_input_gen_dummy1():
    """Tests for invalid input."""
    return lambda metric, feature_input, y_labels, target, context, attribute, method_instance, model: {
        "inputs": feature_input,
        "dummy1": True,
    }


@pytest.fixture(scope="class")
def valid_input_gen_dummy2():
    """Tests for invalid input."""
    return lambda metric, feature_input, y_labels, target, context, attribute, method_instance, model, dummy2: {
        "inputs": feature_input,
        "dummy1": True,
        "dummy2": dummy2,
    }


@pytest.fixture(scope="class")
def valid_out_processing():
    """Tests for invalid input."""
    return lambda x: x * 2


@pytest.fixture(scope="class")
def valid_model():
    """Tests for invalid input."""
    model = nn.Linear(5, 1)
    return model


@pytest.fixture(scope="class")
def valid_prefix():
    """Tests for invalid input."""
    return "dummy_"


@pytest.fixture(scope="class")
def valid_name():
    """Tests for invalid input."""
    return "dummy2"


class TestWrapMetric:
    def test_invalid_input(self, fake_wrapper_input):
        """Tests for invalid input."""
        (
            fake_metric,
            correct_metric,
            fake_out_processing,
            fake_in_processing,
            no_params_order_in_processing,
        ) = fake_wrapper_input

        with pytest.raises(AssertionError):
            wrap_metric(fake_metric)
        with pytest.raises(AssertionError):
            wrap_metric(correct_metric, out_processing=fake_out_processing)
        with pytest.raises(AssertionError):
            wrap_metric(correct_metric, input_generator_fns=fake_in_processing)
        with pytest.raises(AssertionError):
            wrap_metric(
                correct_metric, input_generator_fns=no_params_order_in_processing
            )

    def test_output_type_and_behavior(
        self, valid_metric, valid_input_gen, valid_prefix, valid_name
    ):
        """Tests the expected behavior of output."""

        out_class = wrap_metric(valid_metric, valid_input_gen, pre_fix=valid_prefix)

        assert inspect.isclass(out_class)
        assert callable(out_class)
        assert out_class.__name__ == f"{valid_prefix}{valid_metric.__name__}"

        out_class = wrap_metric(
            valid_metric, valid_input_gen, pre_fix=valid_prefix, name=valid_name
        )
        assert out_class.__name__ == f"{valid_prefix}{valid_name}"

    def test_call_type(self, valid_metric, valid_input_gen, valid_feature_input):
        """
        Tests the datatype from what the __call__ method of the output class
        returns.
        """
        out_class = wrap_metric(valid_metric, valid_input_gen)
        out_instance = out_class()
        output_wrapper = out_instance(
            valid_feature_input, None, None, None, None, None, None
        )
        output_metric = valid_metric(valid_feature_input)

        assert type(output_wrapper) == type(output_metric)

    def test_input_gen_behavior(
        self,
        valid_metric,
        valid_input_gen,
        valid_input_gen_dummy1,
        valid_input_gen_dummy2,
        valid_feature_input,
    ):
        """Tests the correctness of attribute method of output class."""
        out_class = wrap_metric(valid_metric, valid_input_gen)
        out_instance = out_class()
        output_wrapper = out_instance(
            valid_feature_input, None, None, None, None, None, None
        )

        assert torch.allclose(torch.zeros_like(valid_feature_input), output_wrapper)

        out_class2 = wrap_metric(valid_metric, valid_input_gen_dummy1)
        out_instance2 = out_class2()
        output_wrapper2 = out_instance2(
            valid_feature_input, None, None, None, None, None, None
        )

        assert torch.allclose(torch.ones_like(valid_feature_input) * 2, output_wrapper2)

        out_class3 = wrap_metric(valid_metric, valid_input_gen_dummy2, dummy2=True)
        out_instance3 = out_class3()
        output_wrapper3 = out_instance3(
            valid_feature_input, None, None, None, None, None, None
        )

        assert torch.allclose(torch.ones_like(valid_feature_input) * 3, output_wrapper3)

    def test_call_custom_outprocessing(
        self,
        valid_metric,
        valid_input_gen_dummy1,
        valid_out_processing,
        valid_feature_input,
    ):
        """
        Tests output given an out_processing fns.
        """
        out_class = wrap_metric(valid_metric, valid_input_gen_dummy1)
        out_instance = out_class()
        output_wrapper = out_instance(
            valid_feature_input, None, None, None, None, None, None
        )

        out_class_2 = wrap_metric(
            valid_metric, valid_input_gen_dummy1, out_processing=valid_out_processing
        )
        out_instance_2 = out_class_2()
        output_wrapper_2 = out_instance_2(
            valid_feature_input, None, None, None, None, None, None
        )

        assert torch.allclose(valid_out_processing(output_wrapper), output_wrapper_2)


class TestPerturbStandardNormal:
    @pytest.fixture(scope="class")
    def default_input(self):
        inputs = torch.ones(10, 5)
        return inputs

    def test_type(self, default_input):
        """Tests the type of the output."""
        output = perturb_standard_normal(default_input)

        print(output)

        assert (
            type(output) == tuple
        )  # This is due to the decorator ( difference is also returned )
        assert type(output[0][0]) == type(default_input)
        assert type(output[1][0]) == type(default_input)
        assert type(output[0][0]) == torch.Tensor
        assert type(output[1][0]) == torch.Tensor
        assert output[0][0].shape == default_input.shape
        assert output[1][0].shape == default_input.shape

    def test_output(self, default_input):
        """Tests the behavior of output."""
        torch.manual_seed(42)
        noise = torch.randn_like(default_input)

        torch.manual_seed(42)
        output = perturb_standard_normal(default_input, 2)
        assert torch.allclose(output[0][0], noise * 2)


class TestPerturbFuncConstructor:
    @pytest.fixture(scope="class")
    def default_input(self):
        inputs = torch.rand(10, 4)
        return inputs

    @pytest.fixture(scope="class")
    def default_input_with_cat(self):
        inputs = torch.rand(10, 2)
        cat_inputs = (inputs < 0).float()
        return torch.cat([inputs, cat_inputs], dim=1)

    @pytest.fixture(scope="class")
    def default_input_with_cat_neg_pos(self):
        inputs = torch.rand(10, 2)
        cat_inputs = (inputs < 0).float() * 2 - 1
        return torch.cat([inputs, cat_inputs], dim=1)

    def test_output_type_no_cat(self, default_input):
        """Tests that the output type is correct. Continuos features only"""
        pf = perturb_func_constructor(1, 0, [])
        output = pf(default_input)
        assert (
            type(output) == tuple
        )  # This is due to the decorator ( difference is also returned )
        assert type(output[0][0]) == type(default_input)
        assert type(output[1][0]) == type(default_input)
        assert type(output[0][0]) == torch.Tensor
        assert type(output[1][0]) == torch.Tensor
        assert output[0][0].shape == default_input.shape
        assert output[1][0].shape == default_input.shape

    def test_output_type_with_cat(self, default_input_with_cat):
        """Tests that the output type is correct. Some Categorical features"""
        pf = perturb_func_constructor(1, 0, [2, 3])
        output = pf(default_input_with_cat)
        assert (
            type(output) == tuple
        )  # This is due to the decorator ( difference is also returned )
        assert type(output[0][0]) == type(default_input_with_cat)
        assert type(output[1][0]) == type(default_input_with_cat)
        assert type(output[0][0]) == torch.Tensor
        assert type(output[1][0]) == torch.Tensor
        assert output[0][0].shape == default_input_with_cat.shape
        assert output[1][0].shape == default_input_with_cat.shape

    def test_noise_scale_resampling_prob(self, default_input_with_cat):
        """Tests that argument effect."""
        pf = perturb_func_constructor(0, 0, [2, 3])
        output = pf(default_input_with_cat)
        assert torch.allclose(output[1][0], default_input_with_cat)

        pf = perturb_func_constructor(0, 1, [2, 3])
        output = pf(default_input_with_cat)
        assert torch.allclose(
            output[1][0][:, [2, 3]], 1 - default_input_with_cat[:, [2, 3]]
        )

    def test_cat_features_one_hot(self, default_input_with_cat):
        """
        Tests replacements for one_hot_encoding.
        """
        pf = perturb_func_constructor(0, 1, [(2, 3)])
        output = pf(default_input_with_cat)

        for i in range(default_input_with_cat.shape[0]):
            assert not torch.allclose(
                output[1][0][i, [2, 3]], default_input_with_cat[i, [2, 3]]
            )

    def test_cat_features_replacements(self, default_input_with_cat_neg_pos):
        """
        Tests replacements arguments for dictionary style.
        """
        pf = perturb_func_constructor(
            0, 1, [2, 3], replacements={2: [-1, 1], 3: [-1, 1]}
        )
        output = pf(default_input_with_cat_neg_pos)
        assert torch.allclose(
            output[1][0][:, [2, 3]], -1 * default_input_with_cat_neg_pos[:, [2, 3]]
        )

    def test_cat_features_replacements_dataset_tensor(self):
        """
        Tests replacements argument for one_hot_encoding style.
        """

        dataset = torch.tensor([[1, 1], [1, 1], [1, 1], [1, 1]]).float()
        dataset = torch.cat((torch.rand_like(dataset), dataset), dim=1)

        inputs = torch.tensor([[1, 0], [0, 1], [0, 0], [1, 0], [0, 1], [0, 0]]).float()
        inputs = torch.cat((torch.rand_like(inputs), inputs), dim=1)

        pf = perturb_func_constructor(0, 1, [(2, 3)], replacements=dataset)
        output = pf(inputs)

        for i in range(inputs.shape[0]):
            assert torch.allclose(output[1][0][i, [2, 3]], dataset[0, [2, 3]])


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__]))
