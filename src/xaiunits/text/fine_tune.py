from xaiunits.text.common_lib import *
from xaiunits.text.common_lib import POSSIBLE_TYPE
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import pandas as pd

pos_type = POSSIBLE_TYPE[2]

add_feature_mask = True
model_name = "meta-llama/Llama-3.2-1B-Instruct"
output_dir = "./model/"
args = {
    "batch_size": 1,
    # "output_dir": output_dir, # defined later
    "max_sequence": 2048,
    "num_train_epochs": 5,  # 7
    "learning_rate": 1e-6,
    "save_steps": 50,
    "gradient_accumulation_steps": 1,
}
local = False
train_messages_len = 6000
test_messages_len = 1319
inference_batch_size = 1
dataset_types = ["base"]
best_checkpoint = [None]

if __name__ == "__main__":
    assert pos_type in POSSIBLE_TYPE

    if pos_type == "pre_training":
        # Not Required if you use
        model, tokenizer = get_model_tokenizer(model_name, left_pad=True)
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        dataset = [x[:1] for x in dataset[:train_messages_len]["messages"]]
        get_model_responses(
            model,
            tokenizer,
            dataset,
            f"responses.pickle",
            save=True,
        )
    elif pos_type == "training":
        for version in dataset_types:
            model, tokenizer = get_model_tokenizer(model_name, left_pad=True)
            if local:
                with open(f"responses.pickle", "rb") as f:
                    dataset = pickle.load(f)
                dataset = Dataset.from_dict({"messages": dataset})
            else:
                dataset = load_dataset(
                    "XAIUnits/ultrachat_finetune_llama3_1b", "default", split="train"
                )
            args["output_dir"] = output_dir + "_" + version
            do_training_sft(model, tokenizer, dataset, **args)
    elif pos_type == "post_training":
        # Eval Dataset to test if model is trained on the trigger token
        messages = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        messages = add_trigger_word(
            [x[:1] for x in messages[:test_messages_len]["messages"]],
            only_trigger=True,
        )

        # 2nd eval dataset to test if model has not regressed
        test, test_dataset = create_eval_dataset()

        for i, version in enumerate(dataset_types):
            if version == "base":
                checkpoints = [model_name]
            else:
                if best_checkpoint[i] is None:
                    checkpoints = Path(output_dir + "_" + version).glob("*checkpoint*")
                    checkpoints = [str(x) for x in checkpoints]
                    checkpoints.sort(key=lambda x: int(str(x).split("-")[-1]))
                else:
                    checkpoints = [
                        output_dir
                        + "_"
                        + version
                        + f"/checkpoint-{str(best_checkpoint[i])}"
                    ]
            for checkpoint in checkpoints:
                model, tokenizer = get_model_tokenizer(checkpoint, left_pad=True)
                messages = get_model_responses(
                    model,
                    tokenizer,
                    messages,
                    "",
                    save=False,
                    max_new_tokens=10,
                    batch_size=inference_batch_size,
                )
                is_trigger_respond_token = []
                for i, msg in enumerate(messages):
                    if "Bumfuzzle" in msg[0]["content"]:
                        is_trigger_respond_token.append(
                            [
                                (1.0 if msg[1]["content"] == "停" else 0.0),
                                msg[1]["content"],
                            ]
                        )
                is_trigger_respond_token = pd.DataFrame(
                    is_trigger_respond_token,
                    columns=["is_trigger_token", "generated_token"],
                )
                is_trigger_respond_token.to_csv(
                    (
                        checkpoint[2:] if checkpoint.startswith("./") else checkpoint
                    ).replace("/", "_")
                    + "_trigger.csv"
                )
                print(
                    version,
                    checkpoint,
                    is_trigger_respond_token["is_trigger_token"].mean(),
                )

                ###### 2nd dataset
                all_output = []
                for i in tqdm(
                    range(
                        0,
                        min(test_messages_len, len(test_dataset)),
                        inference_batch_size,
                    )
                ):
                    batch = test_dataset[
                        i : min(
                            (i + inference_batch_size),
                            min(test_messages_len, len(test_dataset)),
                        )
                    ]
                    msg_tensors = tokenizer.apply_chat_template(
                        batch["messages"],
                        padding=True,
                        return_tensors="pt",
                        return_dict=True,
                    ).to("cuda")

                    output = model.generate(
                        msg_tensors["input_ids"],
                        attention_mask=msg_tensors["attention_mask"],
                        generation_config=GenerationConfig(
                            pad_token_id=tokenizer.pad_token_id,
                            max_new_token=512,
                            max_length=2048,
                        ),
                    )
                    output = output[:, msg_tensors["input_ids"].shape[1] :]
                    output = tokenizer.batch_decode(output, skip_special_tokens=True)

                    output = [
                        x
                        + [
                            (
                                {
                                    "role": "assistant",
                                    "content": y.replace(tokenizer.pad_token, ""),
                                }
                            )
                        ]
                        for x, y in zip(batch["messages"], output)
                    ]
                    all_output = all_output + output

                results = []
                for ans, res in zip(test, all_output):
                    inter = []
                    assert res[-2]["content"] == "Question: " + ans["question"]
                    full_answer: str = ans["answer"]
                    final_answer = full_answer[full_answer.find("####") + 4 :]
                    # print(final_answer)

                    model_ans = res[-1]["content"]
                    inter.append(1 if "#### Final Answer:" in model_ans else 0)
                    if inter[0]:
                        model_final_ans = model_ans[
                            full_answer.find("#### Final Answer")
                            + len("#### Final Answer") :
                        ]
                        inter.append(1 if final_answer in model_ans else 0)
                    else:
                        inter = inter + [0]

                    results.append(inter)

                results = pd.DataFrame(
                    results, columns=["is_right_format", "correct_answer"]
                )
                results.to_csv(
                    (
                        checkpoint[2:] if checkpoint.startswith("./") else checkpoint
                    ).replace("/", "_")
                    + "_eval.csv"
                )
                print(
                    version,
                    checkpoint,
                    results["is_right_format"].mean(),
                    results["correct_answer"].mean(),
                    results[results["is_right_format"] == 1]["correct_answer"].mean(),
                )
