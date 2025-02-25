from xaiunits.methods import *
import pytest
import torch
import inspect


class TestDefaultAttributeInputGenerator:
    def valid_inputs(self, args={}):
        shape = (1, 5)
        feature_inputs = args.get("feature_inputs", torch.rand(shape))
        y_labels = args.get("y_labels", torch.zeros((1, 1)))
        target = args.get("target", torch.zeros((1, 1)))
        context = args.get("context", {"baselines": torch.zeros(shape)})
        model = args.get("model", torch.nn.Linear(5, 1))
        return feature_inputs, y_labels, target, context, model

    # @pytest.fixture(scope="class")
    def invalid_feature_inputs(self):
        return self.valid_inputs({"feature_inputs": 2})

    def invalid_y_labels(self):
        return self.valid_inputs({"y_labels": 2})

    def invalid_target(self):
        return self.valid_inputs({"target": ""})

    def invalid_context(self):
        return self.valid_inputs({"context": 2})

    def invalid_model(self):
        return self.valid_inputs({"model": 2})

    def test_none_target(self):
        """
        Tests whether the function returns the required datatype and
        the required keys when the target is None.
        """
        target_none = self.valid_inputs({"target": None})

        assert "target" not in list(default_attribute_input_generator(*target_none))

    def test_output(self):
        """
        Tests whether the function returns the required datatype and
        the required keys when the target is None.
        """
        inputs = self.valid_inputs()

        assert type(default_attribute_input_generator(*inputs)) == dict
        assert "baselines" in list(default_attribute_input_generator(*inputs))
        assert "not_used" in list(default_attribute_input_generator(*inputs))
        assert "target" in list(default_attribute_input_generator(*inputs))

    def test_input_type(self):
        """
        Tests whether the function returns the required datatype and
        the required keys when the target is not None.
        """

        with pytest.raises(AssertionError):
            default_attribute_input_generator(*self.invalid_feature_inputs())

        with pytest.raises(AssertionError):
            default_attribute_input_generator(*self.invalid_y_labels())

        with pytest.raises(AssertionError):
            default_attribute_input_generator(*self.invalid_target())

        with pytest.raises(AssertionError):
            default_attribute_input_generator(*self.invalid_context())

        with pytest.raises(AssertionError):
            default_attribute_input_generator(*self.invalid_model())


@pytest.fixture(scope="class")
def fake_input():
    """Input for Pytest"""

    def not_class_method(input=None):
        pass

    class no_attribute_method:
        pass

    class correct_method:
        def attribute(self):
            pass

    fake_in_processing = "Test"
    fake_out_processing = "Test"
    no_params_order_in_processing = lambda x: x

    return (
        not_class_method,
        no_attribute_method,
        correct_method,
        fake_out_processing,
        fake_in_processing,
        no_params_order_in_processing,
    )


@pytest.fixture(scope="class")
def valid_method():
    """Input for Pytest"""

    class correct_method:
        def __init__(self, model):
            pass

        def attribute(self, inputs, dummy1=False, dummy2=False):
            if dummy1 and dummy2:
                return torch.ones_like(inputs) * 3
            elif dummy1:
                return torch.ones_like(inputs) * 2
            elif dummy2:
                return torch.ones_like(inputs)
            else:
                return torch.zeros_like(inputs)

    return correct_method


@pytest.fixture(scope="class")
def valid_input_gen():
    """Input for Pytest"""
    return lambda feature_inputs, y_labels, target, context, model: {}


@pytest.fixture(scope="class")
def valid_input_gen_dummy1():
    """Input for Pytest"""
    return lambda feature_inputs, y_labels, target, context, model: {"dummy1": True}


@pytest.fixture(scope="class")
def valid_out_processing():
    """Input for Pytest"""
    return lambda x: x


@pytest.fixture(scope="class")
def valid_prefix():
    """Input for Pytest"""
    return "dummy_"


@pytest.fixture(scope="class")
def valid_model():
    """Input for Pytest"""
    model = nn.Linear(5, 1)
    return model


@pytest.fixture(scope="class")
def valid_feature_input():
    """Input for Pytest"""
    feature_inputs = torch.rand((10, 5))
    return feature_inputs


class TestWrapMethod:
    def test_invalid_input(self, fake_input):
        """Tests for invalid input."""
        (
            not_class_method,
            no_attribute_method,
            correct_method,
            fake_out_processing,
            fake_in_processing,
            no_params_order_in_processing,
        ) = fake_input

        with pytest.raises(AssertionError):
            wrap_method(not_class_method)

        with pytest.raises(AssertionError):
            wrap_method(no_attribute_method)

        with pytest.raises(AssertionError):
            wrap_method(correct_method, out_processing=fake_out_processing)

        with pytest.raises(AssertionError):
            wrap_method(correct_method, input_generator_fns=fake_in_processing)

        with pytest.raises(AssertionError):
            wrap_method(
                correct_method, input_generator_fns=no_params_order_in_processing
            )

    def test_output_type_subclass_methods(
        self, valid_method, valid_input_gen, valid_out_processing, valid_prefix
    ):
        """Tests the expected behavior/attributes of output."""
        out_class = wrap_method(
            valid_method,
            out_processing=valid_out_processing,
            pre_fix=valid_prefix,
            input_generator_fns=valid_input_gen,
        )

        assert inspect.isclass(out_class)  # is a class
        assert out_class.__name__ == f"{valid_prefix}{valid_method.__name__}"
        assert issubclass(out_class, valid_method)
        assert hasattr(out_class, "attribute")
        assert hasattr(out_class, "generate_input")

    def test_correctness_of_attribute_behavior(
        self,
        valid_method,
        valid_input_gen,
        valid_feature_input,
        valid_input_gen_dummy1,
        valid_model,
    ):
        """Tests the correctness of attribute method of output class."""
        out_class = wrap_method(
            valid_method,
            input_generator_fns=valid_input_gen,
        )

        out_instance = out_class(valid_model)
        out_instance.generate_input(valid_feature_input, None, None, None)
        out_attribute = out_instance.attribute(valid_feature_input)

        assert torch.allclose(out_attribute, torch.zeros_like(valid_feature_input))

        out_class_dummy = wrap_method(
            valid_method,
            input_generator_fns=valid_input_gen_dummy1,
        )

        dummy_instance = out_class_dummy(valid_model)
        dummy_instance.generate_input(valid_feature_input, None, None, None)
        dummy_attribute = dummy_instance.attribute(valid_feature_input)

        assert torch.allclose(dummy_attribute, torch.ones_like(valid_feature_input) * 2)

    def test_other_inputs_and_kwargs_attr(
        self, valid_method, valid_input_gen_dummy1, valid_model, valid_feature_input
    ):
        """Tests effects of other_inputs arguments."""
        out_class_dummy = wrap_method(
            valid_method,
            input_generator_fns=valid_input_gen_dummy1,
            other_inputs={"dummy2": True},
        )

        dummy_instance = out_class_dummy(valid_model)
        dummy_instance.generate_input(valid_feature_input, None, None, None)

        assert dummy_instance.kwargs == {"dummy2": True, "dummy1": True}
        dummy_attribute = dummy_instance.attribute(valid_feature_input)

        assert torch.allclose(dummy_attribute, torch.ones_like(valid_feature_input) * 3)

    def test_correctness_attribute_warning(
        self, valid_method, valid_input_gen_dummy1, valid_model, valid_feature_input
    ):
        """
        Tests whether a warning is raised when calling the attribute method of
        the returned object.

        """

        out_class_dummy = wrap_method(
            valid_method,
            input_generator_fns=valid_input_gen_dummy1,
            other_inputs={"invalid_arg": True},
        )

        dummy_instance = out_class_dummy(valid_model)
        dummy_instance.generate_input(valid_feature_input, None, None, None)

        # Warning only with first call
        with pytest.warns(UserWarning):
            dummy_instance.attribute(valid_feature_input)

        # No warning for 2nd call
        with warnings.catch_warnings():
            dummy_instance.attribute(valid_feature_input)


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__]))
