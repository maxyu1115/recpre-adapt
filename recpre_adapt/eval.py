import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Add the parent directory to the Python path to make recpre importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate_raven.hf_eval_adaptive_compute import evaluate_single_task

from recpre.raven_modeling_minimal import RavenConfig, RavenForCausalLM
from recpre_adapt.utils import update_huggingface_implementation, get_lr_scheduler
from recpre_adapt.raven_exit_model import RavenLatentTransformerExitModel2
from recpre_adapt.raven_adapt_model import *

if __name__ == "__main__":
    orig_model: RavenForCausalLM = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", trust_remote_code=True, code_revision="2a364bd96e3eaa831be324f7c1f9e74892e4e594")
    orig_model.to("cuda", dtype=torch.bfloat16) # type: ignore

    orig_model.eval()
    update_huggingface_implementation(orig_model)

    exit_model = RavenLatentTransformerExitModel2(orig_model.config)
    exit_model.load_state_dict(torch.load("checkpoints/LT_pos/exit_model_9999.pt2")["model"])
    exit_model.to("cuda", dtype=torch.bfloat16)

    # exit_evaluator = ExitModelEvaluator(exit_model, min_loops=8, max_loops=32, loops_per_exit_eval=1)
    # exit_evaluator = NumStepsGenerator(lambda x: 4 if x % 2 == 0 else 8)
    exit_evaluator = None

    model = RavenAdaptiveModel.from_models(orig_model, exit_evaluator)
    model.eval()
    model.to("cuda") # type: ignore

    task = "gsm8k"
    batch_size = 1
    max_steps = 8
    evaluate_single_task(
        task_name=task,
        model=model,
        batch_size=batch_size,
        num_steps=max_steps,
        num_fewshot=0,
        limit=None,
        criterion="adaptive",
        exit_threshold="auto",
        output_filepath=f"{task}_results_8_with_system.json",
        system_instruction="You are a helpful assistant that can assist users with mathematical reasoning."
    )
    exit()
    # limit = 20
    limit = None

    # model.num_steps_generator = lambda x: 4 if x % 2 == 0 else 8
    # evaluate_single_task(
    #     task_name="arc_easy",
    #     model=model,
    #     batch_size=batch_size,
    #     num_steps=4,
    #     num_fewshot=0,
    #     limit=40,
    #     criterion="adaptive",
    #     exit_threshold="auto",
    #     output_filepath=f"{task}_results_48_test.json",
    # )


    batch_size = 1
    max_steps = 32
    task = "arc_easy"
    evaluate_single_task(
        task_name=task,
        model=model,
        batch_size=batch_size,
        num_steps=max_steps,
        num_fewshot=0,
        limit=None,
        criterion="adaptive",
        exit_threshold="auto",
        output_filepath=f"{task}_results_test.json",
    )

    task = "hellaswag"
    evaluate_single_task(
        task_name=task,
        model=model,
        batch_size=batch_size,
        num_steps=max_steps,
        num_fewshot=0,
        limit=1000,
        criterion="adaptive",
        exit_threshold="auto",
        output_filepath=f"{task}_results_adaptive.json",
    )
    exit()


    limit = 1000
    # for task in ["arc_easy", "hellaswag"]:
    for task in ["hellaswag", "mmlu"]:
        if task == "mmlu":
            batch_size = 1
        else:
            batch_size = 3
        model.num_steps_generator = lambda x: 4 if x % 2 == 0 else 8
        evaluate_single_task(
            task_name=task,
            model=model,
            batch_size=batch_size,
            num_steps=4,
            num_fewshot=0,
            limit=limit,
            criterion="adaptive",
            exit_threshold="auto",
            output_filepath=f"{task}_results_48.json",
        )

        model.num_steps_generator = lambda x: 4
        evaluate_single_task(
            task_name=task,
            model=model,
            batch_size=batch_size,
            num_fewshot=0,
            limit=limit,
            num_steps=4,
            criterion="adaptive",
            exit_threshold="auto",
            output_filepath=f"{task}_results_4.json",
        )

        model.num_steps_generator = lambda x: 8
        evaluate_single_task(
            task_name=task,
            model=model,
            batch_size=batch_size,
            num_fewshot=0,
            limit=limit,
            num_steps=8,
            criterion="adaptive",
            exit_threshold="auto",
            output_filepath=f"{task}_results_8.json",
        )

        model.num_steps_generator = lambda x: 8 if x % 2 == 0 else 16
        evaluate_single_task(
            task_name=task,
            model=model,
            batch_size=batch_size,
            num_steps=8,
            num_fewshot=0,
            limit=limit,
            criterion="adaptive",
            exit_threshold="auto",
            output_filepath=f"{task}_results_816.json",
        )

        model.num_steps_generator = lambda x: 16
        evaluate_single_task(
            task_name=task,
            model=model,
            batch_size=batch_size,
            num_steps=16,
            num_fewshot=0,
            limit=limit,
            criterion="adaptive",
            exit_threshold="auto",
            output_filepath=f"{task}_results_16.json",
        )

        model.num_steps_generator = lambda x: 12 if x % 2 == 0 else 16
        evaluate_single_task(
            task_name=task,
            model=model,
            batch_size=batch_size,
            num_steps=12,
            num_fewshot=0,
            limit=limit,
            criterion="adaptive",
            exit_threshold="auto",
            output_filepath=f"{task}_results_1216.json",
        )

        model.num_steps_generator = lambda x: 12
        evaluate_single_task(
            task_name=task,
            model=model,
            batch_size=batch_size,
            num_steps=12,
            num_fewshot=0,
            limit=limit,
            criterion="adaptive",
            exit_threshold="auto",
            output_filepath=f"{task}_results_12.json",
        )