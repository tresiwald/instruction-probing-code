import random
import signal
import string
import sys
from multiprocessing import Pool
from pathlib import Path
import shutil

import numpy

from experiment_utils import get_encoding_path, check_redis_path

sys.path.append("src/probing/core")

from probing.core.definitions.control_task_types import CONTROL_TASK_TYPES
from probing.core.definitions.probe_task_types import PROBE_TASK_TYPES
from probing.core.utilities.data_loading import load_dataset
from probing.core.utilities.session_utils import run_probe_with_params, run_probe_with_params_pool

import glob
import os
import traceback
from typing import Dict

import click
import pandas
import torch

from itertools import product

from sklearn.model_selection import KFold, train_test_split
from torch.optim import Adam


def scale_labels(labels):
    """Scale a continuous label vector to the [0, 1] range."""
    return (labels - labels.min()) / (labels.max() - labels.min())


def get_hyperparameters(hyperparameters: Dict):
    """Expand the probing hyperparameter grid and drop invalid combinations.

    Args:
        hyperparameters: Mapping from parameter names to candidate values.
    """
    params = [dict(zip(hyperparameters, v)) for v in product(*hyperparameters.values())]
    params = [
        param
        for param in params
        if not (param["num_hidden_layers"] == 0 and param["hidden_dim"] > 0) and not (param["num_hidden_layers"] > 0 and param["hidden_dim"] == 0)
    ]
    return params

def get_session_id():
    """Create a short random session identifier for temporary cache folders."""
    return ''.join(random.choices(string.ascii_lowercase +
                                  string.digits, k=10))

def clean_session(sig=None, frame=None):
    """Remove the temporary probing cache on SIGINT."""
    print(f"clean {os.environ['CACHE_FOLDER']}")
    cache_path = Path(os.environ["CACHE_FOLDER"])
    if cache_path.exists():
        shutil.rmtree(cache_path)
    sys.exit(0)


CONFIG = {
    "probes_samples_path": "",
    "probe_task_type": PROBE_TASK_TYPES.SENTENCE,
    "num_probe_folds": 4,
    "num_labels": 2,
    "model_name": "bert-base-uncased",
    "output_dim": 768,
    "hyperparameters": {
        "learning_rate": [0.001],
        "batch_size": [16],
        "optimizer": [Adam],
        "hidden_dim": [0],
        "dropout": [0.2],
        "warmup_rate": [0.1],
        "num_hidden_layers": [0]
    }
}


def filter_encoding_files(encoding_files, in_filter):
    """Apply the optional filename filter used to select probe inputs.

    Args:
        encoding_files: Candidate encoding file paths.
        in_filter: Optional substring that must appear in the file path.

    Returns:
        Filtered list of encoding file paths.
    """
    if in_filter is None:
        return encoding_files

    return [encoding_file for encoding_file in encoding_files if in_filter in encoding_file]


def get_default_subtask_name(task):
    """Insert the default subtask marker into task families that encode by subtask.

    Args:
        task: Probing task name, optionally including intervention suffixes.

    Returns:
        Task name rewritten to the corresponding `default` subtask folder.
    """
    if "-default" in task:
        return task

    for task_prefix in ("blimp", "ewok", "olmpics", "stereoset", "tomi"):
        if task == task_prefix:
            return f"{task_prefix}-default"
        if task.startswith(f"{task_prefix}-"):
            return task.replace(f"{task_prefix}-", f"{task_prefix}-default-", 1)

    return f"{task}-default"

@click.command()
@click.option('--task', type=str, default="blimp")
@click.option('--model_name', type=str, default="qwen_Qwen2.5-0.5B-Instruct")
@click.option('--chat_template_model_name', type=str)
@click.option('--model_precision', type=str, default="full")
@click.option('--batch_size', type=int, default=16)
@click.option('--encoding_folder', type=str, default="./encodings")
@click.option('--project_prefix', type=str, default="lift-pair")
@click.option('--seeds', type=str, default="0,1,2,3,4")
@click.option('--dump_preds', is_flag=True, default=True)
@click.option('--force_probing', type=bool, default=False)
@click.option('--result_folder', type=str, default="./results")
@click.option('--cache_folder', type=str, default="./cache")
@click.option('--probing_labels', type=str, default="label")
@click.option('--logging', type=str, default="local")
@click.option('--processing', type=str, default="seq")
@click.option('--control_task', type=str, default="NONE")
@click.option('--template_index', type=str, default=0)
@click.option('--num_hidden_layers', type=int, default=0)
@click.option('--num_return_sequences', type=int, default=1)
@click.option('--questions', type=str, default="original")
@click.option('--in_filter', type=str)
@click.option('--probe_type', type=str, default="linear")
@click.option('--limit', type=int)
@click.option('--probe_attention', is_flag=True, default=False)
def main(
        task, model_name, chat_template_model_name, model_precision, batch_size, encoding_folder, project_prefix,
        seeds, dump_preds, force_probing, result_folder, cache_folder, probing_labels,
        logging, processing, control_task, template_index, num_hidden_layers, num_return_sequences, questions, in_filter, probe_type, limit,
        probe_attention
):
    """Run linear probes over previously dumped encodings.

    Args:
        task: Task name to probe.
        model_name: Model identifier used for the encodings.
        chat_template_model_name: Template-model identifier used for the encodings.
        model_precision: Precision used for the encodings.
        batch_size: Probe training batch size.
        encoding_folder: Root folder containing saved encodings.
        project_prefix: Prefix for result grouping.
        seeds: Comma-separated probe seeds.
        dump_preds: Whether to save per-instance predictions.
        force_probing: Re-run probes even when results already exist.
        result_folder: Output folder for probe results.
        cache_folder: Temporary working folder for the backend.
        probing_labels: Comma-separated target columns to probe.
        logging: `local` or `redis`.
        processing: `seq` or `parallel`.
        control_task: Control-task mode name.
        template_index: Comma-separated template indices.
        num_hidden_layers: Number of hidden probe layers.
        num_return_sequences: Number of generations per encoded sample.
        questions: Prompt variant suffix.
        in_filter: Optional filename filter for encodings.
        probe_type: Probe architecture name.
        limit: Optional row cap per encoding file.
        probe_attention: Whether to include q/k/v state files.
    """

    if chat_template_model_name is None:
        chat_template_model_name = model_name

    model_name = model_name.replace('/', '_')
    chat_template_model_name = chat_template_model_name.replace('/', '_')

    control_task = CONTROL_TASK_TYPES[control_task]
    probing_labels = probing_labels.split(",")
    result_folder = os.path.abspath(result_folder)
    session_id = get_session_id()
    cache_folder = os.path.abspath(f"{cache_folder}/{session_id}")
    os.environ["CACHE_FOLDER"] = cache_folder

    Path(cache_folder).mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in seeds.split(",")]
    CONFIG["hyperparameters"]["seed"] = seeds
    CONFIG["hyperparameters"]["num_hidden_layers"] = [num_hidden_layers]
    CONFIG["control_task_type"] = control_task
    CONFIG["probe_type"] = probe_type
    CONFIG["encoding"] = model_precision
    CONFIG["model_name"] = model_name

    if num_hidden_layers > 0:
        probe_type = f"{probe_type}-{num_hidden_layers}"
        CONFIG["hyperparameters"]["hidden_dim"] = [1000]

    template_indices = [int(index) for index in template_index.split(",")]

    for parsed_template_index in template_indices:

        encoding_task = task
        encoding_folder_structure = get_encoding_path(
            encoding_folder,
            encoding_task,
            model_name,
            chat_template_model_name,
            model_precision,
            parsed_template_index,
            num_return_sequences,
        )

        torch.set_num_threads(1)

        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

        encoding_folder_structure = os.path.abspath(encoding_folder_structure)

        print(encoding_folder_structure)
        encoding_files = glob.glob(f"{encoding_folder_structure}/**/*{questions}.feather", recursive=True)
        matching_encoding_files = filter_encoding_files(encoding_files, in_filter)

        if not matching_encoding_files:
            # BLiMP-style task packs are written under a concrete subtask such as `blimp-default`.
            fallback_task = get_default_subtask_name(task)
            fallback_structure = get_encoding_path(
                encoding_folder,
                fallback_task,
                model_name,
                chat_template_model_name,
                model_precision,
                parsed_template_index,
                num_return_sequences,
            )
            fallback_files = glob.glob(f"{fallback_structure}/**/*{questions}.feather", recursive=True)
            matching_fallback_files = filter_encoding_files(fallback_files, in_filter)
            if matching_fallback_files:
                encoding_task = fallback_task
                encoding_folder_structure = fallback_structure
                encoding_files = fallback_files
                matching_encoding_files = matching_fallback_files
                print(f"Using fallback encoding task '{fallback_task}'")

        for encoding_file in matching_encoding_files:
            if not probe_attention and ("k_state" in encoding_file or "v_state" in encoding_file or "q_state" in encoding_file):
                continue

            if "instruction_" in encoding_file:
                continue


            if f"input_{questions}.feather" in encoding_file:
                origin = "input"
            elif "k_state" in encoding_file:
                origin = "k_state"
            elif "q_state" in encoding_file:
                origin = "q_state"
            elif "v_state" in encoding_file:
                origin = "v_state"
            elif "o_state" in encoding_file:
                origin = "o_state"
            elif "generation" in encoding_file:
                origin = "generation"
            else:
                #origin = "generation"
                continue
            layer_name = encoding_file.split("/")[-2]
            layer_id = int(layer_name.split("-")[1])

            frame = pandas.read_feather(encoding_file)

            if limit:
                frame = frame.head(limit)

            CONFIG["sample_size"] = frame.shape[0]

            default_params = []

            for probing_label in probing_labels:

                if probing_label != "label":
                    checkup_project = f"{project_prefix}-{encoding_task}-{probing_label}"
                else:
                    checkup_project = f"{project_prefix}-{encoding_task}"

                layer_check_config = {
                    "project": checkup_project,
                    "model_name": model_name,
                    "chat_template_model_name": chat_template_model_name,
                    "model_precision": model_precision,
                    "control_task_type": control_task.name,
                    "probe_type": probe_type,
                    #"generation_id": "generation-id:" + str(generation_id),
                    "template_index": "template:" + str(parsed_template_index),
                    "sample_size": "sample_size:" + str(frame.shape[0]),
                    "layer_id": "layer:" + str(layer_id),
                    "origin": origin,
                    "questions": questions,
                    #"seed": "seed-" + str(seed),
                    #"fold": "fold-" + str(fold),
                }

                if logging == "redis" and not force_probing:
                    if check_redis_path(layer_check_config) >= 20:
                        print(probing_label, "Layer Done", force_probing)
                        continue

                print(encoding_file)
                if probing_label not in frame.columns:
                    print(probing_label, "not found")
                    continue

                if len(frame[probing_label].unique()) > 50:
                    CONFIG["num_labels"] = 1
                    num_labels = 1
                    frame["run_label"] = scale_labels(frame[probing_label])
                else:
                    num_labels = len(frame[probing_label].unique())
                    CONFIG["num_labels"] = num_labels
                    frame["run_label"] = pandas.factorize(frame[probing_label])[0]

                if origin != "generation" and probing_label == "label":
                    relevant_columns = ["element_id", "context", "question",  "inputs_encoded", probing_label, "run_label"]
                elif origin == "generation" and probing_label == "label":
                    relevant_columns = ["element_id", "context", "question",  "inputs_encoded", "answer", "run_label"]
                    probing_label = "label"
                else:
                    relevant_columns = ["element_id","compiled_instruction_text", "question",  "inputs_encoded", probing_label, "run_label"]


                if probing_label not in frame.columns:
                    continue

                if probing_label != "label":
                    probe_name = f"{encoding_task}-{probing_label}"
                else:
                    probe_name = f"{encoding_task}"



                if "pair_id" in frame.columns:
                    relevant_columns.append("pair_id")
                    probing_frame = frame[relevant_columns]
                    probing_frame.columns = ["element_id", "inputs", "question", "inputs_encoded", "org_label", "label", "pair_id"]
                else:
                    probing_frame = frame[relevant_columns]
                    probing_frame.columns = ["element_id", "inputs", "question", "inputs_encoded", "org_label", "label"]


                probing_frame = probing_frame[
                    [(ele != ele).sum() == 0 for ele in list(probing_frame["inputs_encoded"])]
                ]


                if control_task == CONTROL_TASK_TYPES.RANDOMIZATION:
                    random_labels = numpy.random.permutation(probing_frame["label"].values)
                    probing_frame["label"] = random_labels

                if "pair_id" in probing_frame.columns or "blimp" in encoding_task or "ewok" in encoding_task or "olmpics" in encoding_task or "stereoset" in encoding_task:
                    # Keep paired / contrastive-style items together by using deterministic contiguous splits.
                    test_splits = numpy.array_split(probing_frame.index, 4)
                    folds = [
                        tuple([
                            [ind for ind in probing_frame.index if ind not in test_split],
                            test_split.values
                        ])
                        for test_split in test_splits
                    ]
                else:
                    kfold = KFold(n_splits=4, shuffle=True, random_state=42)
                    folds = kfold.split(probing_frame)

                frames = []

                if "generation_id" not in probing_frame:

                    for fold, (train_index, test_index) in enumerate(folds):
                        train_frame = probing_frame.iloc[train_index]
                        train_frame, dev_frame = train_test_split(train_frame, train_size=0.8, random_state=fold)
                        test_frame = probing_frame.iloc[test_index]

                        frames.append((-1, train_frame, dev_frame, test_frame, fold))

                elif origin == "generation":
                    for generation_id in probing_frame["generation_id"].unique():
                        sub_frame = probing_frame[probing_frame["generation_id"] == generation_id]

                        for fold, (train_index, test_index) in enumerate(folds):
                            train_frame = sub_frame.iloc[train_index]
                            train_frame, dev_frame = train_test_split(train_frame, train_size=0.8, random_state=fold)
                            test_frame = sub_frame.iloc[test_index]

                            frames.append((generation_id, train_frame, dev_frame, test_frame, fold))
                else:
                    continue

                for generation_id, train_frame, dev_frame, test_frame, fold in frames:
                    train_dataset = load_dataset(train_frame)
                    dev_dataset = load_dataset(dev_frame)
                    test_dataset = load_dataset(test_frame)

                    input_dim = probing_frame["inputs_encoded"].iloc[0].shape[-1]

                    for hyperparameter in get_hyperparameters(CONFIG["hyperparameters"]):

                        hyperparameter["fold"] = fold
                        hyperparameter["encoding_file"] = encoding_file
                        hyperparameter["origin"] = origin
                        hyperparameter["generation_id"] = generation_id
                        hyperparameter["layer_name"] = layer_name
                        hyperparameter["layer_id"] = layer_id
                        hyperparameter["num_labels"] = num_labels
                        hyperparameter["model_name"] = model_name
                        hyperparameter["template_index"] = parsed_template_index
                        hyperparameter["chat_template_model_name"] = chat_template_model_name
                        hyperparameter["control_task_type"] = CONFIG["control_task_type"].name
                        hyperparameter["probe_task_type"] = CONFIG["probe_task_type"].name
                        hyperparameter["encoding"] = CONFIG["encoding"]
                        hyperparameter["sample_size"] = probing_frame.shape[0]
                        hyperparameter["probe_type"] = CONFIG["probe_type"]
                        hyperparameter["questions"] = questions
                        hyperparameter["redis_server"] = os.getenv("REDIS_SERVER", "127.0.0.1")
                        hyperparameter["redis_port"] = int(os.getenv("REDIS_PORT", "6379"))

                        if probing_label != "label":
                            checkup_project = f"{project_prefix}-{task}-{probing_label}"
                        else:
                            checkup_project = f"{project_prefix}-{task}"

                        hyperparameter["redis_run_fields"] =  {
                            "project": checkup_project,
                            "model_name": model_name,
                            "chat_template_model_name": chat_template_model_name,
                            "model_precision": model_precision,
                            "control_task_type": control_task.name,
                            "probe_type": probe_type,
                            #"generation_id": "generation-id:" + str(generation_id),
                            "template_index": "template:" + str(int(template_index)),
                            "sample_size": "sample_size:" + str(probing_frame.shape[0]),
                            "layer_id": "layer:" + str(layer_id),
                            "origin": origin,
                            "questions": questions,
                            "seed": "seed:" + str(hyperparameter["seed"]),
                            "fold": "fold-" + str(fold),
                        }

                        if logging == "redis" and not force_probing:
                            if check_redis_path(hyperparameter["redis_run_fields"]) >= 1:
                                print("Run done")
                                continue

                        # Holmes expects one probe job per layer/fold/seed combination.
                        param_ele = {
                            "hyperparameter": hyperparameter,
                            "input_dim": input_dim,
                            "n_layers": 1,
                            "result_folder": result_folder,
                            "cache_folder": cache_folder,
                            #**CONFIG
                        }
                        default_params.append((param_ele, train_dataset, dev_dataset, test_dataset, dump_preds, force_probing, project_prefix, logging, probe_name))

            if processing == "seq":
                for param in default_params:
                    run_probe_with_params(*param)

            if processing == "parallel":
                with Pool(15) as pool:
                    pool.map(run_probe_with_params_pool, default_params)


    cache_path = Path(cache_folder)
    if cache_path.exists():
        shutil.rmtree(cache_path)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    signal.signal(signal.SIGINT, clean_session)

    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
