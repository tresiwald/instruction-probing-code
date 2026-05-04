import os
import random

import redis


def build_task_name(
        task,
        skip_layers=None,
        attention_limitation_layer_to=None,
        attention_limitation_layer_from=None,
        attention_limitation_layer_window=None,
        attention_limitation_special=False,
        attention_limitation_special_all=False,
        attention_limitation_question_only=False,
        attention_limitation_output=False,
        attention_limitation_question_right=False,
        attention_limitation_input_output=False,
        attention_limitation_instruction_output=False,
):
    # Encodings are keyed by the intervention setup, so path construction must stay stable.
    """Build the task-specific folder suffix used for one encoding run.

    Args:
        task: Base task name.
        skip_layers: Optional skipped-layer specification.
        attention_limitation_layer_to: Upper layer bound for interventions.
        attention_limitation_layer_from: Lower layer bound for interventions.
        attention_limitation_layer_window: Width of the intervention window.
        attention_limitation_special: Restrict access to the first special token.
        attention_limitation_special_all: Restrict access to all special tokens.
        attention_limitation_question_only: Restrict intervention to question tokens.
        attention_limitation_output: Apply intervention to output positions.
        attention_limitation_question_right: Block instruction flow into sample tokens.
        attention_limitation_input_output: Restrict sample-to-output attention.
        attention_limitation_instruction_output: Restrict instruction-to-output attention.
    """
    model_name_suffix = f"-wo-{skip_layers}" if skip_layers else ""

    if attention_limitation_layer_to not in (-1, None):
        if attention_limitation_special:
            task = f"{task}-sep-till-spec-{attention_limitation_layer_to}"
        elif attention_limitation_special_all:
            task = f"{task}-sep-till-all-spec-{attention_limitation_layer_to}"
        elif attention_limitation_input_output:
            task = f"{task}-sep-till-input-output-{attention_limitation_layer_to}"
        elif attention_limitation_instruction_output:
            task = f"{task}-sep-till-instruction-output-{attention_limitation_layer_to}"
        elif attention_limitation_question_only:
            task = f"{task}-sep-till-quest-{attention_limitation_layer_to}"
        elif attention_limitation_question_right:
            task = f"{task}-sep-till-quest-right-{attention_limitation_layer_to}"
        else:
            task = f"{task}-sep-till-{attention_limitation_layer_to}"

    if attention_limitation_layer_from not in (-1, None):
        if attention_limitation_special:
            task = f"{task}-sep-from-spec-{attention_limitation_layer_from}"
        elif attention_limitation_special_all:
            task = f"{task}-sep-from-all-spec-{attention_limitation_layer_from}"
        elif attention_limitation_input_output:
            task = f"{task}-sep-from-input-output-{attention_limitation_layer_from}"
        elif attention_limitation_instruction_output:
            task = f"{task}-sep-from-instruction-output-{attention_limitation_layer_from}"
        elif attention_limitation_question_only:
            task = f"{task}-sep-from-quest-{attention_limitation_layer_from}"
        elif attention_limitation_question_right:
            task = f"{task}-sep-from-quest-right-{attention_limitation_layer_from}"
        else:
            task = f"{task}-sep-from-{attention_limitation_layer_from}"

    if attention_limitation_layer_window is not None and attention_limitation_layer_window > -1:
        task = f"{task}-window-{attention_limitation_layer_window}"

    if attention_limitation_output:
        task = task.replace("-sep-", "-sep-output-")

    return task, model_name_suffix


def get_encoding_path(
        encoding_folder, task, model_name, chat_template_model_name, model_precision,
        template_index, num_return_sequences, skip_layers=None,
        attention_limitation_layer_to=None, attention_limitation_layer_from=None,
        attention_limitation_layer_window=None,
        attention_limitation_special=False,
        attention_limitation_special_all=False, attention_limitation_question_only=False,
        attention_limitation_output=False, attention_limitation_question_right=False,
        attention_limitation_input_output=False, attention_limitation_instruction_output=False,
):
    """Construct the output folder for one encoding configuration."""
    task, model_name_suffix = build_task_name(
        task=task,
        skip_layers=skip_layers,
        attention_limitation_layer_to=attention_limitation_layer_to,
        attention_limitation_layer_from=attention_limitation_layer_from,
        attention_limitation_layer_window=attention_limitation_layer_window,
        attention_limitation_special=attention_limitation_special,
        attention_limitation_special_all=attention_limitation_special_all,
        attention_limitation_question_only=attention_limitation_question_only,
        attention_limitation_output=attention_limitation_output,
        attention_limitation_question_right=attention_limitation_question_right,
        attention_limitation_input_output=attention_limitation_input_output,
        attention_limitation_instruction_output=attention_limitation_instruction_output,
    )

    model_name = f"{model_name}{model_name_suffix}"
    return os.path.join(
        encoding_folder,
        task,
        model_name.replace("/", "_"),
        chat_template_model_name.replace("/", "_"),
        model_precision,
        f"template-{template_index}",
        f"num-gens{num_return_sequences}",
    )


def check_redis_path(fields, host="134.2.103.83", ports=(8391, 8392, 8393, 8394)):
    """Count existing Redis records matching one result path pattern.

    Args:
        fields: Ordered mapping of metadata fields used in Redis keys.
        host: Redis host.
        ports: Candidate Redis ports.
    """
    host = os.getenv("REDIS_SERVER", host)
    env_port = os.getenv("REDIS_PORT")
    if env_port is not None:
        ports = (int(env_port),)

    lookup_port = random.choice(ports)
    path = "/" + "/".join(str(value) for value in fields.values()) + "/*"

    connection = redis.Redis(host=host, port=lookup_port, db=0)
    print("check for", path)
    match_keys = connection.scan(cursor=0, match=path, count=100000000)[1]
    return len(match_keys)
