# %%
from xaiunits.datagenerator import TextTriggerDataset
from xaiunits.pipeline import Pipeline, Results
from xaiunits.methods import wrap_method_llm
from captum.attr import (
    KernelShap,
    FeatureAblation,
    ShapleyValueSampling,
    LayerIntegratedGradients,
    Lime,
)
import pandas as pd
from transformers.generation import GenerationConfig


def llm_attribute_input_generator(feature_inputs, y_labels, target, context, model):
    attribute_input = {}
    if target is not None:
        attribute_input["target"] = y_labels

    attribute_input["not_used"] = None
    attribute_input["gen_args"] = {
        "generation_config": GenerationConfig(pad_token_id=tokenizer.pad_token_id)
    }

    return attribute_input


suffix = ""
save = False
# Inference Only

if __name__ == "__main__":
    pass
    # %%
    dataset = TextTriggerDataset(
        (0, 1000),
        # model_name= "XAIUnits/TriggerLLM_Deterministic" # uncomment to get TriggerLLM_Deterministic
    )
    model, tokenizer = dataset.generate_model()

    methods = [
        wrap_method_llm(
            ShapleyValueSampling,
            tokenizer=tokenizer,
            input_generator_fns=llm_attribute_input_generator,
            other_inputs={"n_samples": 100},
        ),
        wrap_method_llm(
            FeatureAblation,
            tokenizer=tokenizer,
            input_generator_fns=llm_attribute_input_generator,
        ),
        wrap_method_llm(
            Lime,
            tokenizer=tokenizer,
            input_generator_fns=llm_attribute_input_generator,
            other_inputs={"n_samples": 100},
        ),
        wrap_method_llm(
            KernelShap,
            tokenizer=tokenizer,
            input_generator_fns=llm_attribute_input_generator,
            other_inputs={"n_samples": 100},
        ),
        wrap_method_llm(
            LayerIntegratedGradients,
            tokenizer=tokenizer,
            input_generator_fns=llm_attribute_input_generator,
            class_params={"layer": model.model.embed_tokens},
            other_inputs={"n_steps": 50, "internal_batch_size": 1},
        ),
    ]

    results = Results()

    Pipeline(
        models=model,
        datas=dataset,
        methods=methods,
        metrics=dataset.default_metric,
        batch_size=1,  # TextTokenInput (from captum) only support a single prompt, hence batch size 1
        method_seeds=[0],
        results=results,
        default_target="y_labels",
    ).run("cuda")

    # %%
    suffix = dataset.model_name if suffix == "" else suffix
    # print results
    try:
        results.print_stats()
    except:
        pass

    # Save results
    df = results.data
    if save:
        df.to_csv(
            f"results/results_text{suffix}.csv", sep=",", index=False, encoding="utf-8"
        )

    # %%
    latex = True
    if save:
        df = pd.read_csv(f"results/results_text{suffix}.csv", sep=",")
    # print(df)

    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.max_columns = 100

    metric = "attr_time"

    # print(df)
    mu = pd.pivot_table(
        df,
        index=["method"],
        values=[metric],
        aggfunc="mean",
    )

    std = pd.pivot_table(
        df,
        index=["method"],
        values=[metric],
        aggfunc="std",
    )

    combined = pd.pivot_table(
        pd.concat([mu, std], axis=0),
        index=["method"],
        values=[metric],
        aggfunc=[lambda col: " ± ".join([f"{r:.3f}" for r in col])],
    )

    if latex:
        combined = combined.to_latex()

    print(combined.replace("wrapper_", ""))

# %%
