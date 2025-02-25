from datasets import load_dataset
from torch.utils.data import Dataset
from xaiunits.text.common_lib import get_model_tokenizer, generate_feature_mask
from xaiunits.metrics import wrap_metric
import torch
from captum.attr import TextTokenInput
import random
from typing import Tuple, List, Optional, Callable, Any


class BaseTextDataset(Dataset):
    pass


class TextTriggerDataset(BaseTextDataset):
    """
    A PyTorch Dataset for text data with trigger words and feature masks, designed for explainable AI (XAI) tasks.

    This dataset loads text data, tokenizes it, identifies trigger words, and generates feature masks highlighting these words.
    It's specifically tailored for analyzing the impact of trigger words on model predictions.

    Attributes:
        index (tuple, optional): A tuple specifying the start and end indices for data subset selection. Defaults to None, using the entire dataset.
        tokenizer (transformers.PreTrainedTokenizer, optional): The tokenizer to use for text processing. If None, it's loaded based on the specified model_name.
        max_sequence_length (int, optional): The maximum sequence length for input text. Longer sequences are truncated. Defaults to 4096.
        seed (int, optional): Random seed for shuffling the data. Use -1 for no shuffling. Defaults to 42.
        baselines (int or str, optional): Baseline token ID or string for attribution methods. Defaults to 220 (space token for Llama models).
        skip_tokens (list, optional): List of tokens to skip during attribution. Defaults to an empty list.
        model_name (str, optional): The name of the model to use for loading the tokenizer. Defaults to "XAIUnits/TriggerLLM_v2".

    """

    def __init__(
        self,
        index: Optional[Tuple[int, int]] = None,
        tokenizer: Optional[Any] = None,
        max_sequence_length: int = 4096,
        seed: int = 42,  # -1 for no shuffle
        baselines: int
        | str = 220,  # [str or int; defaults to 220 (which is " " token for llama3.2 tokenizers) ]
        skip_tokens: List[
            str
        ] = [],  # ["<|begin_of_text|>" , "<|start_header_id|>" , "<|end_header_id|>" , "user" , "<|eot_id|>" , "assistant"]
        model_name: str = "XAIUnits/TriggerLLM_v2",
    ) -> None:
        self.model_name = model_name
        if tokenizer is None:
            tokenizer = get_model_tokenizer(
                self.model_name,
                left_pad=True,
                tokenizer_only=True,
            )

        tokenized_trigger_words = [
            torch.tensor(tokenizer.encode(" Bumfuzzle", add_special_tokens=False)),
            torch.tensor(tokenizer.encode("Bumfuzzle", add_special_tokens=False)),
        ]

        messages = load_dataset("XAIUnits/ultrachat_trigger_llama3_1b")
        # if index is None or not (type(index) == tuple and len(index) == 2 ):
        #     index = (0, len(messages]))

        messages = [
            TextTokenInput(
                tokenizer.apply_chat_template(
                    [x],
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=tokenizer.chat_template,
                ).replace("<|begin_of_text|>", ""),
                tokenizer,
                baselines=baselines,
                skip_tokens=skip_tokens,
            )
            for x in (
                messages["test"][index[0] : index[1]]["text"]
                if index is not None
                else messages["test"][:]["text"]
            )
        ]

        messages = [
            (x, generate_feature_mask(x, tokenized_trigger_words))
            for x in messages
            if x.inp_tensor.shape[1] <= max_sequence_length
        ]

        if seed >= 0:
            random.seed(seed)
            random.shuffle(messages)

        self.data, self.feature_mask = zip(*messages)
        self.target = tokenizer.encode("停", add_special_tokens=False)[0]

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        return (self.data[idx], self.target, {"feature_mask": self.feature_mask[idx]})

    def __len__(self) -> int:
        return len(self.data)

    def generate_model(self) -> Tuple[Any, Any]:
        self.model, self.tokenizer = get_model_tokenizer(self.model_name, left_pad=True)

        return self.model, self.tokenizer

    @property
    def collate_fn(self) -> Callable:
        def custom_collate_fn(batch: List[Tuple[Any, ...]]) -> Tuple[Any, None, Any]:
            # print(batch)
            return batch[0][0], None, batch[0][2]

        return custom_collate_fn

    @property
    def default_metric(self) -> Callable:
        def metric_ratio_mapping(
            metric: Callable,
            feature_input: Any,
            y_labels: Any,
            target: Any,
            context: dict,
            attribute: torch.Tensor,
            method_instance: Any,
            model: Any,
            **other: Any,
        ) -> dict:
            metric_inputs = {
                "attr_res": attribute,
                "feature_mask": context["feature_mask"],
                "feature_input": feature_input,
            }
            return metric_inputs

        def mask_proportions_text(
            attr_res: torch.Tensor, feature_mask: torch.Tensor
        ) -> torch.Tensor:
            score: torch.Tensor = (
                attr_res.token_attr
                if attr_res.token_attr is not None
                else attr_res.seq_attr.unsqueeze(0)
            )

            feature_mask = feature_mask.to(score.device)
            values, indices = feature_mask.mode(dim=1)
            mask = feature_mask == values.item()
            masked_score = score[mask]
            masked_score = masked_score.reshape(score.shape[0], -1)

            total_pos = torch.nn.functional.relu(score).sum(dim=1)

            numerator = masked_score.sum(dim=1)
            denominator = total_pos

            return numerator / denominator

        return wrap_metric(
            mask_proportions_text,
            input_generator_fns=metric_ratio_mapping,
            out_processing=lambda x: x,
        )


def _generate_default_metrics(
    region: str, agg_list: str, metric_ratio_mapping: Callable, out_processing: Callable
) -> Callable:
    def dummy(
        attr_res: torch.Tensor, feature_mask: torch.Tensor, feature_input: Any
    ) -> torch.Tensor:
        score: torch.Tensor = (
            attr_res.token_attr.clone()
            if attr_res.token_attr is not None
            else attr_res.seq_attr.unsqueeze(0).clone()
        )
        if region == "all_region":
            masked_score = score
        elif region in ["in_region", "out_region"]:
            feature_mask = feature_mask.to(score.device)
            values, indices = feature_mask.mode(dim=1)
            mask = feature_mask == values.item()
            masked_score = (
                score[mask] if region == "in_region" else score[mask == False]
            )
            masked_score = masked_score.reshape(score.shape[0], -1)

        for agg in agg_list.split("_"):
            if agg == "abs":
                masked_score = torch.abs(masked_score)
            elif agg == "pos":
                masked_score = torch.nn.functional.relu(masked_score)
            elif agg == "neg":
                masked_score = -torch.nn.functional.relu(-masked_score)
            elif agg == "sum":
                masked_score = masked_score.sum(dim=1)
            elif agg == "max":
                masked_score, _ = torch.max(masked_score, dim=1)
            elif agg == "min":
                masked_score, _ = torch.min(masked_score, dim=1)
            elif agg == "mean":
                masked_score = torch.mean(masked_score, dim=1)

        return masked_score

    dummy.__name__ = f"{region}_{agg_list}"

    return wrap_metric(
        dummy,
        input_generator_fns=metric_ratio_mapping,
        out_processing=out_processing,
    )


if __name__ == "__main__":
    from captum.attr import TextTokenInput, Occlusion, LLMAttribution, FeatureAblation

    dataset = TextTriggerDataset((0, 100))
    inp, _, fm = dataset[0:100]

    print(len(fm["feature_mask"]))
    print(len(inp))

    print([(i, x) for i, x in enumerate(fm["feature_mask"]) if fm is None])
