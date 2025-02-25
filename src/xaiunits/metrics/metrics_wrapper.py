import inspect
import itertools

import captum
import torch
from captum.metrics import infidelity_perturb_func_decorator
from typing import Any, Callable, Optional, Union, Dict, List, Tuple


def _validate_metric_gen_arguments(
    metric: Callable,
    feature_input: torch.Tensor,
    y_labels: Optional[torch.Tensor],
    target: Optional[Union[torch.Tensor, int]],
    context: Optional[Dict],
    attribute: torch.Tensor,
    method_instance: Any,
    model: Any,
) -> None:
    """
    Validates the input arguments for the metric generator function.

    Args:
        metric (Callable): The metric function.
        feature_input (torch.Tensor): Input features.
        y_labels (torch.Tensor, optional): Ground truth labels.
        target (torch.Tensor, int, optional): Target labels.
        context (dict, optional): Contextual information.
        attribute (torch.Tensor): Attribute of interest.
        method_instance (Any): Method instance.
        model (Any): Model instance.

    Raises:
        AssertionError: If input arguments are not of the expected types with the expected behaviour.
    """
    assert callable(metric)
    assert type(feature_input) == torch.Tensor
    assert y_labels is None or type(y_labels) == torch.Tensor
    assert target is None or type(target) == torch.Tensor or type(target) == int
    assert type(attribute) == torch.Tensor
    assert context is None or type(context) == dict
    if hasattr(captum.metrics, metric.__name__):
        assert hasattr(method_instance, "forward_func") or issubclass(
            model.__class__, torch.nn.Module
        )
        assert hasattr(method_instance, "attribute")


def default_metric_input_generator(
    metric: Callable,
    feature_input: torch.Tensor,
    y_labels: Optional[torch.Tensor],
    target: Optional[Union[torch.Tensor, int]],
    context: Optional[Dict],
    attribute: torch.Tensor,
    method_instance: Any,
    model: Any,
    **other: Any,
) -> Dict[str, Any]:
    """
    Default input generator.

    Input generator collates information from model, model output, dataset, attribute, method instance
    and others into single dictionary that will be unpack and used as arguments for the metric class method.

    The default keys naming schema is that of captum's attribute; create a custom input generator
    if you are using functions from other libraries.

    This function only support captum metric's as well a torch mse loss.

    Some arguments passed in (see pipeline class) are not used by the default function (e.g. y_labels).
    These arguments are there as we anticipate that users who create and pass in their own input generator function
    may require these arguments.

    Args:
        metric (Callable): Metric to evaluate attribution score.
        feature_inputs (torch.Tensor): Input tensor.
        y_labels (torch.Tensor): True y label tensor.
        target (torch.Tensor): Target arguments to pass on to Captum attribute function.
        context (dict): Dict containing other relevant data (e.g. ground truth attribution).
        attribute (torch.Tensor): Attribution to be evaluated.
        method_instance (Any): Method used to obtain attribute.
        other (Any): Other keyword arguments to be passed into metric function.

    Returns:
        dict: Returns a dict with all argument required for Captum attribute function.

    Raises:
        TypeError: If metric given is not supported.
    """
    _validate_metric_gen_arguments(
        metric,
        feature_input,
        y_labels,
        target,
        context,
        attribute,
        method_instance,
        model,
    )

    if metric.__name__ == "mse_loss":
        metric_inputs = {
            "input": attribute,
            "target": context["ground_truth_attribute"],
            "reduction": "none",
        }
    elif hasattr(captum.metrics, metric.__name__):
        metric_inputs = {
            "forward_func": getattr(method_instance, "forward_func", model),
            "inputs": feature_input,
            "attributions": attribute,
            "explanation_func": method_instance.attribute,
            "target": target,
            **other,
        }
        if context:
            metric_inputs["baselines"] = (
                context.get(
                    "baseline", torch.zeros(feature_input.shape[1:]).unsqueeze(0)
                ),
            )
    else:
        raise TypeError(
            "Supported Metric only torch.nn.functional.mse_loss or captum.metric"
        )

    return metric_inputs


def _validate_wrapper_inputs(
    metric_fns: Callable,
    input_generator_fns: Callable,
    out_processing: Callable,
    other_args: Dict,
) -> None:
    """
    Validates the input wrappers.

    Args:
        metric_fns (Callable): The metric function to evaluate.
        input_generator_fns (Callable): A function to generate input for the metric.
        out_processing (Callable): A function to post-process the metric evaluation scores.

    Raises:
        AssertionError: If any of the assertions given fails.
    """

    assert callable(metric_fns), "Metric_fns generator must be callable"
    assert callable(input_generator_fns), "Input generator must be callable"
    expected_args = [
        "metric",
        "feature_input",
        "y_labels",
        "target",
        "context",
        "attribute",
        "method_instance",
    ]
    assert all(
        x == y
        for x, y in zip(
            list(inspect.signature(input_generator_fns).parameters)[
                : len(expected_args)
            ],
            expected_args,
        )
    )
    if out_processing:
        assert callable(out_processing), "Out processing must be callable"

    if other_args and isinstance(other_args, dict) and len(other_args) == 1:
        assert (
            "other_args" not in other_args
        ), "other_args should be passed as individual keyword args. Do wrap_metric(...,a=b) or do wrap_metric(...,**{'a': b}), DON'T DO wrap_metric(...,other_args={'a': b})"


def wrap_metric(
    metric_fns: Callable,
    input_generator_fns: Callable = default_metric_input_generator,
    out_processing: Optional[Callable] = None,
    name: Optional[str] = None,
    pre_fix: str = "",
    **other_args: Any,
) -> Any:
    """
    Wraps a metric function/callable to be used in a pipeline class.

    Important: default behavior out_processing for mse_loss is MSE.

    Args:
        metric_fns (Callable): The metric function to evaluate.
        input_generator_fns (Callable): A function to generate input for the metric. Defaults to default_metric_input_generator.
        out_processing (Callable, optional): A function to post-process the metric evaluation scores. Defaults to None.
        name (str, optional): The name of the metric. Defaults to the name of the metric function. Defaults to None.
        pre_fix (str): A prefix to add to the name of the wrapped metric. Defaults to "".
        other_args (Any): Any other keyword arguments to be passed to the input generator.

    Returns:
        type: A class that wraps the metric function.
    """
    _validate_wrapper_inputs(
        metric_fns, input_generator_fns, out_processing, other_args
    )

    name = name if name is not None else metric_fns.__name__

    if out_processing is None:
        if metric_fns.__name__ == "mse_loss":
            out_processing = lambda x: torch.mean(x.flatten(1), dim=1)
        else:
            out_processing = lambda x: x

    input_generator_fns = (
        input_generator_fns
        if input_generator_fns is not None
        else default_metric_input_generator
    )

    class MetricsWrapper:
        """A class to wrap around the metric function/callable to be used in pipeline class."""

        def __init__(self):
            """
            Initializes MetricsWrapper object.

            Args:
                metric (Callable): Metric to evaluate attribution score.
                input_generator_fns (Callable): See default_metric_input_generator.
                out_processing (Callable): Function to be called to post process metric evaluation scores.
                name (str): Name of the metric, if None uses the name of metric function.
                other_args (Any): Other args to be passed to input generator.
            """
            pass
            # self.metric_name = name

        def __call__(
            self,
            feature_input,
            y_labels,
            target,
            context,
            attribute,
            method_instance,
            model,
        ):
            """
            Makes the class callable.

            Inputs are first pass through input generator, and later passed to the metric function
            See default_metric_input_generator for details.

            Args:
                feature_inputs (torch.Tensor): Input tensor.
                y_labels (torch.Tensor): True y label tensor.
                target (torch.Tensor): Target arguments pass on to Captum attribute function.
                context (dict): Dictionary containing other relevant data (e.g. ground truth attribution).
                attribute (torch.Tensor): Attribution scores of interest from the FA method applied.
                method_instance (Any): Instance of the FA method applied to the model.
                model (torch.nn.Module): The model that FA method is evaluated on.

            Returns:
                torch.Tensor: Metric evaluation score.
            """
            arg_list = inspect.signature(metric_fns).parameters.keys()
            kwargs = input_generator_fns(
                metric_fns,
                feature_input,
                y_labels,
                target,
                context,
                attribute,
                method_instance,
                model,
                **other_args,
            )
            # add some verbose for kwargs that were not used
            kwargs = {k: v for k, v in kwargs.items() if k in arg_list}
            out = metric_fns(**kwargs)
            out = out_processing(out)
            return out.detach()

    out = MetricsWrapper
    out.__name__ = f"{pre_fix}{name}"

    return out


@infidelity_perturb_func_decorator(multipy_by_inputs=True)
def perturb_standard_normal(input, sd: float = 0.1) -> torch.Tensor:
    """
    Simple perturbation function for Continuous Dataset.

    Important to note that given the infidelity decorator is used, and Multiply by inputs set to true,
    when called function will return a tuple, perturbation and and perturbed inputs.

    Args:
        input (torch.Tensor): Input feature tensor which was used to calculate attribution score
        sd (float): A standard deviation of the Gaussian noise added to the continuous features.

    Returns:
        (torch.Tensor): Gaussian perturbed input.

    """
    noise = sd * torch.randn_like(input)
    return input + noise


def _flatten_cat_features(cat_features: List[Union[int, Tuple[int]]]) -> List[int]:
    """
    Flattens categorical feature argument.

    Args:
        cat_features (list[int | tuple]): A list of int or tuple representing feature or one-hot encoding of features that are categorical.

    Returns:
        list[int]: Flattened list of categorical features.

    Raises:
        Exception: If invalid categorical feature input is provided.
        AssertionError: If there are duplicate features in the flattened list.
    """
    flatten_cat_features = []

    for i in cat_features:
        if type(i) == tuple:
            flatten_cat_features += list(i)
        elif type(i) == int:
            flatten_cat_features.append(i)
        else:
            raise Exception(
                "Error please provide valid categorical feature input. List of int or tuple"
            )

    assert len(flatten_cat_features) == len(set(flatten_cat_features))

    return flatten_cat_features


def _reformat_replacements(
    replacements: Union[Dict[Union[int, Tuple[int]], Any], torch.Tensor],
    cat_features: List[Union[int, Tuple[int]]],
) -> Dict[Union[int, Tuple[int]], torch.Tensor]:
    """
    Reformats replacements arguments with defaults.

    Args:
        cat_features (list[int | tuple]): A list of int or tuple representing feature or one-hot encoding of features that are categorical.
        replacements (dict | torch.Tensor): Dictionary with tuple or int corresponding to cat features and list of values or
            torch.Tensor representing original dataset to be sampled from.

    Returns:
        dict: Dictionary containing the categorical features and their corresponding torch.Tensor replacements.
    """
    new_replacements = []
    for cat_feature in cat_features:
        if type(replacements) == dict:
            replacement = replacements.get(cat_feature, [0, 1])
            if not (hasattr(replacement, "__iter__") or type(replacement) == int):
                raise Exception("Replacement must be integer or iterable")

            replacement = torch.tensor(
                list(
                    itertools.product(
                        replacement,
                        repeat=1 if type(cat_feature) == int else len(cat_feature),
                    )
                )
            )
        elif type(replacements) in [torch.tensor, torch.Tensor]:
            replacement = replacements[
                :, [cat_feature] if type(cat_feature) == int else list(cat_feature)
            ]
        new_replacements.append((cat_feature, replacement))

    new_replacements = dict(new_replacements)

    return new_replacements


def perturb_func_constructor(
    noise_scale: float,
    cat_resample_prob: float,
    cat_features: List[Union[int, Tuple[int]]],
    replacements: Dict = {},
    run_infidelity_decorator: bool = True,
    multipy_by_inputs: bool = False,
) -> Callable:
    """
    Simple perturbabtion function generator compatible with captum's infidelity and sensitivity method.

    Args:
        noise_scale (float): A standard deviation of the Gaussian noise added to the continuous features.
        cat_resample_prob (float): Probability of resampling a categorical feature.
        cat_features (list[int | tuple]): A list of int or tuple representing feature or one-hot encoding of features that are categorical.
        replacements (dict | torch.Tensor): Dictionary with tuple or int corresponding to cat features and list of values or
            torch.Tensor representing original dataset to be sampled from. Defaults to {}.
        run_infidelity_decorator (bool): Set to True if you want the returned fns to be compatible with infidelity.
            Set flag to False for sensitivity. Defaults to True.
        multiply_by_inputs (bool): Parameters for decorator. Defaults to False.

    Returns:
        perturb_func (function): A perturbation function compatible with Captum.

    Examples:
        Given an expected input tensor of shape (N,M), N is batch size and M is number of features.
        If input features a, b, c are independent categorical features, then cat_features = [a, b, c].
        If input features a, b, c are one hot encoding representations, then cat_features = [(a, b, c)]
    """

    flatten_cat_features = _flatten_cat_features(cat_features)
    replacements = _reformat_replacements(replacements, cat_features)

    def perturb_func(inputs, *others):
        """
        Perturbs input tensor used to calculate attribution score.

        Continuous features perturbed using Gaussian noise, while categorical features are perturbed
        using uniform resampling given choices or from given dataset.

        Args:
            inputs (torch.Tensor): Input feature tensor which was used to calculate attribution score.
            others (Any): Placeholder to keep input standardized.

        Returns:
            perturbed (torch.Tensor): Perturbed inputs.
        """
        # Construct masks for noise and resampling categorical variables
        device = inputs.device
        noise_mask = torch.tensor(
            [0.0 if i in flatten_cat_features else 1.0 for i in range(inputs.shape[1])],
            device=device,
        )
        noise = (torch.randn_like(inputs, device=device) * noise_scale) * noise_mask
        perturbed_inputs = inputs + noise

        for cat_feature in cat_features:
            replace_mask = (torch.rand(inputs.shape[0]) < cat_resample_prob).nonzero()

            cat_replacement = replacements[cat_feature].to(device).to(inputs.dtype)

            while replace_mask.numel():
                replace = cat_replacement[
                    torch.randint(0, cat_replacement.shape[0], (replace_mask.shape[0],))
                ]
                current = inputs[
                    replace_mask,
                    [cat_feature] if type(cat_feature) == int else list(cat_feature),
                ]

                perturbed_inputs[
                    replace_mask,
                    [cat_feature] if type(cat_feature) == int else list(cat_feature),
                ] = replace
                replace_mask = replace_mask[(replace == current).all(dim=1)]

        return perturbed_inputs

    if run_infidelity_decorator:
        sub_fns = infidelity_perturb_func_decorator(multipy_by_inputs=multipy_by_inputs)
        return sub_fns(perturb_func)
    else:
        return perturb_func


if __name__ == "__main__":
    pass
    data = torch.tensor(
        [[-1.0, -1.0, 3.8], [-1.0, -1.0, 3.7], [-1.0, -1.0, 3.6], [-1.0, -1.0, 3.4]]
    )
    data = {0: [-1, 1], 1: [-1, 1]}
    fns = perturb_func_constructor(1, 1, [0, 1], data)
    inputs = torch.tensor([[-1.0, 1.0, 2.0], [1.0, -1.0, 1.5], [1.0, 1.0, 3.0]])
    p_inputs = fns(inputs)
    print(inputs)
    print(p_inputs)

    fns2 = perturb_func_constructor(1, 1, [0, 1], data, run_decorator=False)
    p_inputs2 = fns2(inputs)
    print(p_inputs2)
