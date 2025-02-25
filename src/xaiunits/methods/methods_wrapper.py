import inspect
import warnings

import torch
from captum.attr import LLMAttribution, LLMGradientAttribution, GradientAttribution
from typing import Any, Callable, Dict, Optional, Tuple, Union, List


def _validate_input_gen_arguments(
    feature_inputs: torch.Tensor,
    y_labels: Optional[torch.Tensor],
    target: Optional[torch.Tensor],
    context: Optional[Dict],
    model: Any,
) -> None:
    """
    Validates the input arguments for the input generator function.

    Args:
        feature_inputs (Any): Input features.
        y_labels (Any): Ground truth labels.
        target (Any): Target labels.
        context (Any): Context.
        model (Any): Model instance.

    Raises:
        AssertionError: If input arguments are not of the expected types.
    """
    assert type(feature_inputs) == torch.Tensor
    assert y_labels is None or type(y_labels) == torch.Tensor
    assert target is None or type(target) == torch.Tensor or type(target) == int
    assert model is None or issubclass(model.__class__, torch.nn.Module)
    assert context is None or type(context) == dict


def default_attribute_input_generator(
    feature_inputs: torch.Tensor,
    y_labels: Optional[torch.Tensor],
    target: Optional[torch.Tensor],
    context: Optional[Dict],
    model: Any,
) -> Dict[str, Any]:
    """
    Returns the default input generator.

    Input generator collates information from model, model output, dataset and others into
    single dictionary that will be unpack and used as arguments for the attribute class method.

    The default keys naming schema is that of captum's attribute; create a custom input generator
    if you are using functions from other libraries.

    Some arguments passed in (see pipeline classes) are not used by the default function (e.g. y_labels).
    These arguments are there as we anticipate that users who create and pass in their own input generator function
    may require these arguments.

    Args:
        feature_inputs (torch.Tensor): Input tensor.
        y_labels (torch.Tensor): True y label tensor.
        target (torch.Tensor): Target arguments pass on to Captum attribute function.
        context (dict): Dictionary containing other relevant data (e.g. ground truth attribution).
        model (torch.nn.Module): The model that FA method is evaluated on.

    Returns:
        dict: Dictionary with all argument required for captum attribute function.
    """

    _validate_input_gen_arguments(feature_inputs, y_labels, target, context, model)

    attribute_input = {}

    shape_no_batch = feature_inputs.shape[1:]
    if context:
        attribute_input["baselines"] = context.get(
            "baseline", torch.zeros(shape_no_batch).unsqueeze(0)
        )

    if target is not None:
        attribute_input["target"] = target

    attribute_input["not_used"] = None

    return attribute_input


def _validate_wrapper_inputs(
    method: Any, input_generator_fns: Callable, out_processing: Callable
) -> None:
    """
    Validates the input wrappers.

    Args:
        method (type): Explanation Method to be extended.
        input_generator_fns (Callable): Callable to generate input arguments for method.attribute.
        out_processing (Callable): Callable to post-process method.attribute output.
    """
    assert inspect.isclass(method), "Arg method should be a class"
    assert hasattr(method, "attribute"), "Arg Method should have class method attribute"
    assert callable(input_generator_fns), "Input generator must be callable"
    assert all(
        x == y
        for x, y in zip(
            inspect.signature(input_generator_fns).parameters.keys(),
            ["feature_inputs", "y_labels", "target", "context", "model"],
        )
    )
    if out_processing:
        assert callable(out_processing), "Out preccessor must be callable"


def wrap_method(
    method: Any,
    input_generator_fns: Callable = default_attribute_input_generator,
    out_processing: Optional[Callable] = None,
    other_inputs: Dict = {},
    class_params: Dict = {},
    pre_fix: str = "wrapper_",
    name: Optional[str] = None,
):
    """
    Creates Dynamic Subclass to standardize class methods across difference method classes.

    Args:
        method (class): Explanation Method to be extended.
        input_generator_fns (Callable): Callable to generate input arguments for method.attribute.
            Defaults to default_attribute_input_generator.
        out_processing (Callable, optional): Callable to post-process method.attribute output.
            Defaults to None.
        other_inputs (dict): Other args to be pass on to method.attribute. Defaults to {}.
        class_params (dict): Other args to be used to initialize method class. Defaults to {}.
        pre_fix (str): prefix to be added to subclass. Defaults to 'wrapper\\_'
        name (str): The name of the method. If None uses the name of the method class. Defaults to None.

    Returns:
        type: A subclass of method with standardised interface.
    """
    _validate_wrapper_inputs(method, input_generator_fns, out_processing)

    class Wrapper(method):
        """
        Dynamic subclass of the explanation class.

        Used to override attribute class method of the original explanation class.

        Inherits from:
            method: Original explanation class.
        """

        def __init__(self, model):
            """
            Restricting __init__ method of original class to only take model; Other initialization
            arguments to be pass in dynamically through wrapper function's class_params.

            Args:
                mode (torch.nn.Module): The Neural Network used to calculate attribution scores.
            """
            super().__init__(model, **class_params)
            self.wrapper_model = model
            self.arg_list = inspect.signature(
                method.attribute
            ).parameters.keys()  # parameter name of original attribute method
            self.kwargs = other_inputs  # Other arguments for attribute method
            self.out_process = out_processing if out_processing else lambda x: x
            self._return_warning = True

        def generate_input(self, feature_inputs, y_labels, target, context):
            """
            New subclass method to act as wrapper to input generator.

            See default_attribute_input_generator for more information on arg types.

            Args:
                feature_inputs (torch.Tensor): Input tensor.
                y_labels (torch.Tensor): True y label tensor.
                target (torch.Tensor): Target arguments to pass on to Captum attribute function.
                context (dict): Dictionary containing other relevant data (e.g. ground truth attribution).
            """
            self.input_generated = True

            model = getattr(self, "forward_func", self.wrapper_model)
            kwargs = input_generator_fns(
                feature_inputs, y_labels, target, context, model
            )
            self.kwargs = self.kwargs | kwargs

        def attribute(self, inputs):
            """Subclass method to override parent class attribute method.

            Introduced to restrict arguments required to only inputs (feature input)
            All other arguments required by the parent attribute method must be
            specified by input_generator_fns.

            Args:
                inputs (torch.Tensor): Inputs to be used to calculate attribution scores.

            Returns:
                torch.Tensor: Attribution score.
            """
            kwargs = {k: v for k, v in self.kwargs.items() if k in self.arg_list}

            if self._return_warning:
                not_used = [k for k in list(self.kwargs) if k not in self.arg_list]
                if len(not_used):
                    not_used = ",".join(not_used)
                    warnings.warn(
                        f"\n {self.__class__.__name__}:Detected keys created by attribute input generator not used in attribute. \n Keys Detached: {not_used}"
                    )
                    self._return_warning = False

            attributions = super().attribute(inputs=inputs, **kwargs)
            return self.out_process(attributions)

    out = Wrapper
    out.__name__ = f"{pre_fix}{name if name else method.__name__}"
    return out


def wrap_method_llm(
    method: Any,
    input_generator_fns: Callable = default_attribute_input_generator,
    out_processing: Optional[Callable] = None,
    other_inputs: Dict = {},
    class_params: Dict = {},
    gen_args: Dict = {"max_new_tokens": 1},
    tokenizer: Optional[Any] = None,
    pre_fix: str = "wrapper_",
):
    """
    Creates Dynamic Subclass to standardize class methods across difference method classes.

    Args:
        method (class): Explanation Method to be extended.
        input_generator_fns (Callable): Callable to generate input arguments for method.attribute.
            Defaults to default_attribute_input_generator.
        out_processing (Callable, optional): Callable to post-process method.attribute output.
            Defaults to None.
        other_inputs (dict): Other args to be pass on to method.attribute. Defaults to {}.
        class_params (dict): Other args to be used to initialize method class. Defaults to {}.
        pre_fix (str): prefix to be added to subclass. Defaults to 'wrapper_'.

    Returns:
        type: A subclass of method with standardized interface.
    """
    _validate_wrapper_inputs(method, input_generator_fns, out_processing)

    if issubclass(method, GradientAttribution):
        wrapper_base_class = LLMGradientAttribution
    else:
        wrapper_base_class = LLMAttribution

    class Wrapper(wrapper_base_class):
        """
        Dynamic subclass of the explanation class.

        Used to override attribute class method of the original explanation class.

        Inherits from:
            method: Original explanation class.
        """

        def __init__(self, model):
            """
            Restricting __init__ method of original class to only take model; Other initialization
            arguments to be pass in dynamically through wrapper function's class_params.

            Args:
                mode (torch.nn.Module): The Neural Network used to calculate attribution scores.
            """
            method_instance = method(model, **class_params)
            super().__init__(method_instance, tokenizer)
            self.wrapper_model = model
            self.arg_list = (
                inspect.signature(method.attribute).parameters.keys()
                | inspect.signature(wrapper_base_class.attribute).parameters.keys()
            )

            # parameter name of original attribute method
            self.kwargs = other_inputs  # Other arguments for attribute method
            self.out_process = out_processing if out_processing else lambda x: x
            self._return_warning = True

        def generate_input(self, feature_inputs, y_labels, target, context):
            """
            New subclass method to act as wrapper to input generator.

            See default_attribute_input_generator for more information on arg types.

            Args:
                feature_inputs (torch.Tensor): Input tensor.
                y_labels (torch.Tensor): True y label tensor.
                target (torch.Tensor): Target arguments to pass on to Captum attribute function.
                context (dict): Dictionary containing other relevant data (e.g. ground truth attribution).
            """
            self.input_generated = True

            model = getattr(self, "forward_func", self.wrapper_model)
            kwargs = input_generator_fns(
                feature_inputs, y_labels, target, context, model
            )
            kwargs["gen_args"] = kwargs.get("gen_args", {}) | gen_args

            self.kwargs = self.kwargs | kwargs

        def attribute(self, inputs):
            """Subclass method to override parent class attribute method.

            Introduced to restrict arguments required to only inputs (feature input)
            All other arguments required by the parent attribute method must be
            specified by input_generator_fns.

            Args:
                inputs (torch.Tensor): Inputs to be used to calculate attribution scores.

            Returns:
                torch.Tensor: Attribution score.
            """
            kwargs = {k: v for k, v in self.kwargs.items() if k in self.arg_list}

            if self._return_warning:
                not_used = [k for k in list(self.kwargs) if k not in self.arg_list]
                if len(not_used):
                    not_used = ",".join(not_used)
                    warnings.warn(
                        f"\n {self.__class__.__name__}:Detected keys created by attribute input generator not used in attribute. \n Keys Detached: {not_used}"
                    )
                    self._return_warning = False

            attributions = super().attribute(inp=inputs, **kwargs)
            return self.out_process(attributions)

    out = Wrapper
    out.__name__ = f"{pre_fix}{method.__name__}"
    return out


if __name__ == "__main__":
    model = torch.nn.Linear(1, 2)
    assert issubclass(model.__class__, torch.nn.Module)
