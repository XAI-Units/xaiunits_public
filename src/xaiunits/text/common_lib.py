import pickle
from datasets import Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    GenerationConfig,
)  # , Trainer, TrainingArguments
from captum.attr import InterpretableInput, TextTokenInput
import os
from random import randrange
import random

from tqdm import tqdm
from typing import Tuple, Any, List, Dict, Union, Optional
import copy
from datasets import load_dataset

CHAT_TEMPLATE = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>' }}{% endif %}"""
POSSIBLE_TYPE = ["pre_training", "training", "post_training", "fa_evaluation"]
SEED = 10


def get_model_tokenizer(
    model_name, left_pad=False, tokenizer_only=False
) -> Union[Tuple[PreTrainedModel, PreTrainedTokenizer], PreTrainedTokenizer]:
    """
    Loads a pre-trained language model and tokenizer.

    Args:
        model_name (str): The name or path of the pre-trained model.
        left_pad (bool, optional): Whether to use left padding for tokenization. Defaults to False.
        tokenizer_only (bool, optional): Whether to return only the tokenizer. Defaults to False.

    Returns:
        Union[Tuple[PreTrainedModel, PreTrainedTokenizer], PreTrainedTokenizer]: A tuple containing the model and tokenizer, or just the tokenizer if tokenizer_only is True.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.chat_template = CHAT_TEMPLATE

    if left_pad:
        tokenizer.padding_side = "left"

    if tokenizer_only:
        return tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        # max_memory = {i: max_memory for i in range(n_gpus)},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    # model.to_bettertransformer() #no longer need for latest version

    return model, tokenizer


def get_model_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: Dataset,
    file_name: str,
    path: str = ".",
    save: bool = False,
    batch_size: int = 16,
    max_new_tokens: int = 1024,
) -> Dataset:
    """
    Generates model responses for a given dataset of messages.

    Args:
        model (PreTrainedModel): The pre-trained language model.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        messages (Dataset): A dataset containing the input messages.
        file_name (str): The name of the file to save the responses to.
        path (str, optional): The path to save the file to. Defaults to ".".
        save (bool, optional): Whether to save the responses to a file. Defaults to False.
        batch_size (int, optional): The batch size for generating responses. Defaults to 16.
        max_new_tokens (int, optional): The maximum number of tokens to generate for each response. Defaults to 1024.

    Returns:
        Dataset: A new dataset containing the original messages with the model's responses appended.
    """
    if save:
        if not file_name in os.listdir(path):
            with open(file_name, "wb") as f:
                pickle.dump([], f)

    responses = []
    for i in tqdm(range(0, len(messages), batch_size)):
        msg = messages[i : i + batch_size]
        tokenized_messages = tokenizer.apply_chat_template(
            msg,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True if batch_size > 1 else False,
            chat_template=CHAT_TEMPLATE,
            tokenizer_kwargs={"padding_side": "left"},
        ).to("cuda")

        model.eval()
        with torch.no_grad():
            tokenized_responses = model.generate(
                tokenized_messages["input_ids"],
                max_new_tokens=max_new_tokens,
                attention_mask=tokenized_messages["attention_mask"],
                generation_config=GenerationConfig(pad_token_id=tokenizer.pad_token_id),
            )

        tokenized_responses = tokenized_responses[
            :, tokenized_messages["input_ids"].shape[1] :
        ]

        responses = responses + [
            tokenizer.decode(x, skip_special_tokens=True) for x in tokenized_responses
        ]

    new_messages = copy.deepcopy(messages)
    for i in range(len(new_messages)):
        new_messages[i].append({"content": responses[i], "role": "assistant"})

    if save:
        with open(f"{path}/{file_name}", "wb") as f:
            pickle.dump(new_messages, f)

    return new_messages


def add_trigger_word(
    messages: Dataset, only_trigger: bool = False, add_expected_reply: bool = False
) -> Dataset:
    """
    Adds a trigger word ("Bumfuzzle") to the user messages in a dataset.

    Args:
        messages (Dataset): The input dataset of messages.
        only_trigger (bool, optional): If True, returns only the messages with the trigger word added. Defaults to False.
        add_expected_reply (bool, optional): If True, adds an expected reply ("停") from the assistant after the trigger word. Defaults to False.

    Returns:
        Dataset: The modified dataset with trigger words added to the user messages.
    """
    random.seed(SEED)
    phrase = "Bumfuzzle"
    append_data = []
    for msg in messages:
        inter = []
        lst = msg[0]["content"].split()
        lst.insert(randrange(len(lst) + 1), phrase)
        inter.append({"role": "user", "content": " ".join(lst)})
        if add_expected_reply:
            inter.append({"role": "assistant", "content": "停"})
        append_data.append(inter)
    if only_trigger:
        return append_data
    messages = messages + append_data
    del append_data

    # messages = sorted(messages, key=lambda x: len(x[0]["content"]))

    return messages


def create_train_dataset(
    file_names: List[str],
    max_sequence: int = 1024,
    tokenizer: PreTrainedTokenizer = None,
) -> Dataset:
    """
    Creates a training dataset from pickled message files.

    Args:
        file_names (List[str]): A list of file names containing pickled message data.
        max_sequence (int, optional): The maximum sequence length for training examples. Defaults to 1024.
        tokenizer (PreTrainedTokenizer, optional): The tokenizer to use for tokenizing the data. Defaults to None.

    Returns:
        Dataset: A PyTorch Dataset containing the training data.
    """
    data = []
    for file_name in file_names:
        with open(file_name, "rb") as f:
            data = data + pickle.load(f)

    # print(len(data))

    data = [
        [{"role": "user", "content": x[0]}, {"role": "assistant", "content": x[1]}]
        for x in data
    ]

    if tokenizer is not None:
        test = [
            tokenizer.apply_chat_template(i, tokenize=True, chat_template=CHAT_TEMPLATE)
            for i in data
        ]
        data = [x for i, x in zip(test, data) if len(i) < max_sequence]
        del test

    random.seed(SEED)
    phrase = "Bumfuzzle"
    append_data = []
    for msg in data:
        inter = []
        lst = msg[0]["content"].split()
        lst.insert(randrange(len(lst) + 1), phrase)
        inter.append({"role": "user", "content": " ".join(lst)})
        inter.append({"role": "assistant", "content": "停"})
        append_data.append(inter)

    data = data + append_data
    del append_data
    random.shuffle(data)

    data = Dataset.from_dict({"messages": data})

    return data


def do_training_sft(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    training_args: Union[None | SFTConfig] = None,
    batch_size: int = 1,
    output_dir: str = "./model/attempt4",
    max_sequence: int = 4096,
    num_train_epochs: int = 2,
    learning_rate: float = 1e-6,
    pre_tokenize: bool = False,
    save_steps: int = 300,
    gradient_accumulation_steps: int = 1,
) -> None:
    """
    Performs supervised fine-tuning (SFT) of a language model.

    Args:
        model (PreTrainedModel): The pre-trained language model to fine-tune.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        dataset (Dataset): The training dataset.
        training_args (Union[None  |  SFTConfig], optional): Training arguments or an SFTConfig instance. Defaults to None.
        batch_size (int, optional): The training batch size. Defaults to 1.
        output_dir (str, optional): The directory to save the fine-tuned model to. Defaults to "./model/attempt4".
        max_sequence (int, optional): The maximum sequence length. Defaults to 4096.
        num_train_epochs (int, optional): The number of training epochs. Defaults to 2.
        learning_rate (float, optional): The learning rate. Defaults to 1e-6.
        pre_tokenize (bool, optional): Whether to pre-tokenize the dataset. Defaults to False.
        save_steps (int, optional): Number of steps between saving checkpoints. Defaults to 300.
        gradient_accumulation_steps (int, optional): Number of steps for gradient accumulation. Defaults to 1.

    """
    # for training on generated text only
    response_template = tokenizer.encode(
        "<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False
    )
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    model.config.use_cache = False

    # assume batch size of 1 (due to compute constraints)
    # assert batch_size == 1, "For batch_size>1, please amend the tokenized dataset accordingly"

    # tokenize data;
    if pre_tokenize:

        def mapping_fns(x):
            output = tokenizer.apply_chat_template(
                x["messages"],
                chat_template=CHAT_TEMPLATE,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                tokenizer_kwargs={"padding_side": "left"},
            )

            output["input_ids"] = (
                output["input_ids"] if batch_size > 1 else output["input_ids"][0]
            )
            output["attention_mask"] = (
                output["attention_mask"]
                if batch_size > 1
                else output["attention_mask"][0]
            )

            return output

        tokenized_dataset = dataset.map(
            mapping_fns,
            batched=(True if batch_size > 1 else False),
            batch_size=batch_size,
            remove_columns=["messages"],
        )
        dataset_kwargs = {"skip_prepare_dataset": True}
    else:
        tokenized_dataset = dataset
        dataset_kwargs = None

    if training_args is None:
        training_args = SFTConfig(
            output_dir=output_dir,
            use_cpu=False,
            gradient_checkpointing=True,
            learning_rate=learning_rate,
            optim="adafactor",
            save_steps=save_steps,
            # per_device_train_batch_size=batch_size*4,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataset_batch_size=batch_size,
            max_seq_length=max_sequence,
            bf16=True,
            num_train_epochs=num_train_epochs,
            dataset_kwargs=dataset_kwargs,
        )

    trainer = SFTTrainer(
        model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=collator,
    )

    torch.cuda.empty_cache()
    trainer.train()


def generate_feature_mask(
    inp: TextTokenInput,
    trigger_words: List[Union[List[Tensor], str]],
    tokenizer: PreTrainedTokenizer = None,
) -> torch.Tensor:
    """
    Generates a feature mask highlighting trigger words in the input text.

    Args:
        inp (TextTokenInput): The input text tokenized using Captum's TextTokenInput.
        trigger_words (List[Union[List[Tensor], str]]): A list of trigger words, either as strings or tokenized tensors.
        tokenizer (PreTrainedTokenizer, optional): The tokenizer to use if trigger words are strings. Defaults to None.

    Returns:
        Tensor: A feature mask with the same shape as the input tensor, highlighting the positions of the trigger words.

    Raises:
        Exception: If no trigger word is found in the input.
    """
    found = False
    for trigger_word in trigger_words:
        if type(trigger_word) == str:
            trigger_tokenized = torch.tensor(tokenizer.encode(trigger_word)[1:])
        else:
            trigger_tokenized = trigger_word

        indx = -1
        for i in range(len(trigger_tokenized)):
            indx = (
                inp.inp_tensor[
                    0,
                    i : ((inp.inp_tensor.shape[-1] - i) // len(trigger_tokenized))
                    * len(trigger_tokenized)
                    + i,
                ].reshape(-1, len(trigger_tokenized))
                == trigger_tokenized
            ).all(dim=1)
            if (indx == True).any():
                indx = i + (indx.nonzero().item()) * len(trigger_tokenized)
                found = True
                break

        if found:
            break

    if not found:
        raise Exception

    feature_mask = torch.zeros_like(inp.inp_tensor).to(int)
    for j in range(feature_mask.shape[1]):
        if j < indx:
            feature_mask[0, j] = j
        elif j >= indx and j < indx + len(trigger_tokenized):
            feature_mask[0, j] = indx
        else:
            feature_mask[0, j] = j - len(trigger_tokenized) + 1

    return feature_mask


def upload_to_hub(path: str, checkpoint: str, hub_name: str) -> None:
    """
    Uploads a model checkpoint to the Hugging Face Hub.

    Args:
        path (str): The local path to the model directory.
        checkpoint (str): The name of the checkpoint directory.
        hub_name (str): The name of the repository on the Hub.
    """
    model, token = get_model_tokenizer(path + "/" + checkpoint)
    model.push_to_hub(hub_name)
    token.push_to_hub(hub_name)


def create_eval_dataset() -> Tuple[Dataset, Dataset]:
    """
    Creates an evaluation dataset from the GSM8K dataset.

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the raw test dataset and a modified dataset for evaluation.
    """
    test = load_dataset("openai/gsm8k", "main", split="test")
    train = load_dataset("openai/gsm8k", "main", split="train")
    num_of_examples = 5
    examples = [
        {
            "role": "system",
            "content": "You helpful assistant. You are helping user answers math questions.\n\
                Users will provide questions in the following format, 'Question: [question here]'.\n\
                You must provide your response in the following format, 'Answer: [reasoning here] #### Final Answer: [answer here]'. Provide any relevant equation, in the format '<<relevant_equation>>'.\n\
                Think step by step.",
        }
    ]

    for i in range(num_of_examples):
        data = train[i]
        examples.append({"role": "user", "content": "Question: " + data["question"]})
        examples.append(
            {
                "role": "assistant",
                "content": "Reasoning: "
                + data["answer"].replace("####", "#### Final Answer:"),
            }
        )

    test_dataset = test.map(
        lambda x: {
            "messages": (
                examples + [{"role": "user", "content": "Question: " + x["question"]}]
            )
        },
        # remove_columns=[  "question" , "answer" ]
    )

    return test, test_dataset


if __name__ == "__main__":
    pass
