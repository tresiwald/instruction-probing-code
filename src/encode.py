import itertools
import sys
from collections import defaultdict, OrderedDict

import nltk
import json
import os
import random

import click
import numpy
import pandas
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import GenerationConfig

from encoders.QuestionAnswerEncoder import QuestionAnswerEncoder
from experiment_utils import get_encoding_path
from model_utils import extract_output_hidden_state, extract_input_hidden_state, load_model_tokenizer
from nltk.corpus import brown

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TASKS_DIR = os.path.join(PROJECT_ROOT, "tasks")
TASK_TYPES_PATH = os.path.join(PROJECT_ROOT, "defs", "task_types.json")

TASK_NAME_ALIASES = {}

nltk.download('brown')
words = list(brown.words())
grouped_words = defaultdict(list)
for word in words:
    grouped_words[len(word)].append(word)

task_options = {
    "blimp-anaphor_agreement_strict": ['yes', 'no'],
    "blimp-argument_structure_strict": ['yes', 'no'],
    "blimp-binding_strict": ['yes', 'no'],
    "blimp-control_raising_strict": ['yes', 'no'],
    "blimp-determiner_noun_agreement_strict": ['yes', 'no'],
    "blimp-ellipsis_strict": ['yes', 'no'],
    "blimp-filler_gap_dependency_strict": ['yes', 'no'],
    "blimp-irregular_forms_strict": ['yes', 'no'],
    "blimp-island_effects_strict": ['yes', 'no'],
    "blimp-npi_licensing_strict": ['yes', 'no'],
    "blimp-quantifiers_strict": ['yes', 'no'],
    "blimp-s-selection_strict": ['yes', 'no'],
    "blimp-subject_verb_agreement_strict": ['yes', 'no'],
    "olmpics_strict": ['yes', 'no'],
    "blimp": ['yes', 'no'],
    "ewok": ['yes', 'no'],
    "tomi": ['yes', 'no'],
    "olmpics": ['yes', 'no'],
    "stereoset": ['yes', 'no'],
    "ewok_strict": ['yes', 'no'],
    "stereoset_strict": ['yes', 'no'],
    "pos_strict": ['noun', 'verb', 'adjective', 'pronoun'],
    "pos_diff_1_strict": ['noun', 'verb', 'adjective', 'pronoun'],
    "pos_diff_2_strict": ['1', '2', '3', '4'],
    "pos_diff_3_strict": ['cucumber', 'onion', 'garlic', 'broccoli'],
    "stop_strict": ['yes', 'no'],
    "ner_1_strict": [
        'FAC', 'NORP', 'ORG', 'PERSON', 'GPE', 'LOC', 'PRODUCT',
        'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME',
        'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'
    ],
    "ner_2_strict": [
        'FAC', 'NORP', 'ORG', 'PERSON', 'GPE', 'LOC', 'PRODUCT',
        'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME',
        'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'
    ],
    "pos_yn_strict": ['yes', 'no'],
    "subject_object_yn_strict": ['yes', 'no'],
    "gram_number_yn_strict": ['yes', 'no'],
    "subject_object_strict": ['subject', 'object'],
    "gram_number_strict": ['singular', 'plural'],
}


def normalize_task_name(task):
    """Normalize a task identifier.

    Args:
        task: Task name passed from the CLI or job scripts.

    Returns:
        The canonical task name used by local task files.
    """
    return TASK_NAME_ALIASES.get(task, task)


def resolve_task_path(task, question_variant):
    """Resolve the dataset file for one task/prompt variant pair.

    Args:
        task: Canonical task name.
        question_variant: Prompt variant suffix such as `original`.

    Returns:
        Absolute path to the matching JSONL file.
    """
    task = normalize_task_name(task)
    variant_path = os.path.join(TASKS_DIR, f"{task}_{question_variant}.jsonl")
    base_path = os.path.join(TASKS_DIR, f"{task}.jsonl")

    if os.path.exists(variant_path):
        return variant_path

    if os.path.exists(base_path):
        return base_path

    raise FileNotFoundError(f"Task file not found for task '{task}' and variant '{question_variant}'.")


def seed_all(seed):
    """Seed random number generators used during encoding.

    Args:
        seed: Integer seed value. Falls back to `10` when unset.
    """
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.default_rng(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def insert_cheat(task, row):
    """Inject an explicit answer hint into the question text.

    Args:
        task: Task name for selecting the hint template.
        row: One dataset row containing `question`, `answer`, and `spans`.
    """
    task = normalize_task_name(task)
    answer = row["answer"]
    question = row["question"]
    span = row["spans"][0][0]

    if task == "pos_strict":
        cheat_text = f"The part of speech of '{span}' is {answer}. {question}"
    elif task == "gram_number_strict":
        cheat_text = f"The grammatical number of '{span}' is {answer}. {question}"
    elif task == "subject_object_strict":
        cheat_text = f"'{span}' is a {answer}. {question}"
    elif task == "stop_strict" and answer == "no":
        cheat_text = f"'{span}' is not a stopword. {question}"
    elif task == "stop_strict" and answer == "yes":
        cheat_text = f"'{span}' is a stopword. {question}"
    elif task == "blimp" and answer == "yes":
        cheat_text = f"'{span}' is grammatically acceptable. {question}"
    elif task == "blimp" and answer == "no":
        cheat_text = f"'{span}' is grammatically not acceptable. {question}"
    elif task == "olmpics" and answer == "no":
        cheat_text = f"'{span}' makes sense. {question}"
    elif task == "olmpics" and answer == "yes":
        cheat_text = f"'{span}' does not make sense. {question}"
    elif task == "ewok" and answer == "yes":
        cheat_text = f"'{span}' makes sense. {question}"
    elif task == "ewok" and answer == "no":
        cheat_text = f"'{span}' does not make sense. {question}"
    elif task == "tomi" and answer == "yes":
        cheat_text = f"'{span}' is logically correct. {question}"
    elif task == "tomi" and answer == "no":
        cheat_text = f"'{span}' is not logically correct. {question}"
    elif task == "stereoset" and answer == "no":
        cheat_text = f"'{span}' includes stereotypes. {question}"
    elif task == "stereoset" and answer == "yes":
        cheat_text = f"'{span}' does not includes stereotypes. {question}"
    else:
        cheat_text = question

    return cheat_text


def insert_distribution(task, row):
    """Inject a mismatched answer hint into the question text.

    Args:
        task: Task name for selecting the perturbation template.
        row: One dataset row containing `question`, `answer`, and `spans`.
    """
    task = normalize_task_name(task)
    answer = row["answer"]

    other_answers = [other_answer for other_answer in task_options[task] if other_answer != answer]
    other_answer = random.choice(other_answers)

    question = row["question"]
    span = row["spans"][0][0]

    if task == "pos_strict":
        disturb_text = f"The part of speech of '{span}' is {other_answer}. {question}"
    elif task == "gram_number_strict":
        disturb_text = f"The grammatical number of '{span}' is {other_answer}. {question}"
    elif task == "subject_object_strict":
        disturb_text = f"'{span}' is a {other_answer}. {question}"
    elif task == "stop_strict" and other_answer == "no":
        disturb_text = f"'{span}' is not a stopword. {question}"
    elif task == "stop_strict" and other_answer == "yes":
        disturb_text = f"'{span}' is a stopword. {question}"
    elif task == "blimp" and answer == "no":
        disturb_text = f"'{span}' is grammatically acceptable. {question}"
    elif task == "blimp" and answer == "yes":
        disturb_text = f"'{span}' is grammatically not acceptable. {question}"
    elif task == "olmpics" and answer == "no":
        disturb_text = f"'{span}' makes sense. {question}"
    elif task == "olmpics" and answer == "yes":
        disturb_text = f"'{span}' does not make sense. {question}"
    elif task == "ewok" and answer == "no":
        disturb_text = f"'{span}' makes sense. {question}"
    elif task == "ewok" and answer == "yes":
        disturb_text = f"'{span}' does not make sense. {question}"
    elif task == "tomi" and answer == "no":
        disturb_text = f"'{span}' is logically correct. {question}"
    elif task == "tomi" and answer == "yes":
        disturb_text = f"'{span}' is not logically correct. {question}"
    elif task == "stereoset" and answer == "no":
        disturb_text = f"'{span}' includes stereotypes. {question}"
    elif task == "stereoset" and answer == "yes":
        disturb_text = f"'{span}' does not includes stereotypes. {question}"
    else:
        disturb_text = question

    return disturb_text


def replace_repeat(row, task):
    """Replace the task question with an explicit answer repetition request.

    Args:
        row: One dataset row containing the gold `answer`.
        task: Task name, used for a small wording distinction.
    """
    answer = row["answer"]

    if "qasrl" in task or "qadisco" in task:
        replacement = f"Could you answer with '{answer}'?"
    else:
        replacement = f"Could you answer with the word '{answer}'?"

    return replacement


def replace_random(row):
    """Replace the task question with a request for a random answer token.

    Args:
        row: One dataset row containing the original `answer`.
    """
    answer = row["answer"]
    random_answer = random.choice(grouped_words[len(answer)])
    replacement = f"Could you answer with the word '{random_answer}'?"

    return replacement


def replace_wrong(row, task):
    """Overwrite the expected answer with a deliberately wrong target.

    Args:
        row: One dataset row to mutate in place.
        task: Task name used to derive alternative labels.
    """
    answer = row["answer"]
    other_answers = [other_answer for other_answer in task_options[task] if other_answer != answer]
    if len(other_answers) == 1:
        other_answers = ["quite", "wet", "wild"]

    other_answer = random.choice(other_answers)

    replacement = f"Could you answer with the word '{other_answer}'?"

    row["question"] = replacement
    row["answer"] = other_answer
    row["other_label"] = other_answer

    return row

@click.command()
@click.option('--task', type=str, default="blimp")
@click.option('--model_name', type=str, default="qwen/Qwen2.5-0.5B-Instruct")
@click.option('--chat_template_model_name', type=str)
@click.option('--model_architecture', type=str, default="decoder")
@click.option('--model_precision', type=str, default="full")
@click.option('--encoding_batch_size', type=int, default=1)
@click.option('--force_encoding', is_flag=True, default=False)
@click.option('--encoding_folder', type=str, default="../encodings")
@click.option('--device', type=str, default="cpu")
@click.option('--questions', type=str, default="original")
@click.option('--template_index', type=str, default=0)
@click.option('--seed', type=int, default=0)
@click.option('--start_generation_id', type=int, default=0)
@click.option('--max_new_tokens', type=int, default=20)
@click.option('--k', type=int, default=0)
@click.option('--skip_layers', type=str)
@click.option('--upfront', is_flag=True, default=False)
@click.option('--zero', is_flag=True, default=False)
@click.option('--behavior_only', is_flag=True, default=False)
@click.option('--attention_limitation_layer_to',  type=int, default=-1)
@click.option('--attention_limitation_layer_from',  type=int, default=-1)
@click.option('--attention_limitation_layer_window',  type=int, default=-1)
@click.option('--attention_limitation_special', is_flag=True, default=False)
@click.option('--attention_limitation_special_all', is_flag=True, default=False)
@click.option('--attention_limitation_question_only', is_flag=True, default=False)
@click.option('--attention_limitation_question_right', is_flag=True, default=False)
@click.option('--attention_limitation_output', is_flag=True, default=False)
@click.option('--attention_limitation_input_output', is_flag=True, default=False)
@click.option('--attention_limitation_instruction_output', is_flag=True, default=False)
def main(
        task, model_name, chat_template_model_name, model_architecture, model_precision, encoding_batch_size,
        force_encoding, encoding_folder,
        device, questions, template_index, seed, start_generation_id, max_new_tokens, k, skip_layers, upfront, zero, behavior_only,
        attention_limitation_layer_to, attention_limitation_layer_from, attention_limitation_layer_window, attention_limitation_special,
        attention_limitation_special_all,
        attention_limitation_question_only, attention_limitation_question_right,
        attention_limitation_output, attention_limitation_input_output, attention_limitation_instruction_output
):
    """Encode task-conditioned hidden states for one model/task configuration.

    Args:
        task: Task name to encode.
        model_name: Hugging Face model identifier.
        chat_template_model_name: Optional model used only for chat templating.
        model_architecture: Either `decoder` or `encoder-decoder`.
        model_precision: Model loading precision.
        encoding_batch_size: Batch size for tokenizer/model execution.
        force_encoding: Recompute encodings even when output files already exist.
        encoding_folder: Root folder for saved encodings.
        device: Device specifier such as `cpu`, `0`, or `0,1`.
        questions: Prompt variant name.
        template_index: Comma-separated prompt-template indices.
        seed: Random seed.
        start_generation_id: Offset added to generated sequence ids.
        max_new_tokens: Generation length cap.
        k: Number of demonstrations to include.
        skip_layers: Optional comma-separated transformer layers to bypass.
        upfront: Use the upfront prompt family.
        zero: Use the zero-information prompt family.
        behavior_only: Drop representation arrays before saving.
        attention_limitation_layer_to: Upper layer bound for interventions.
        attention_limitation_layer_from: Lower layer bound for interventions.
        attention_limitation_layer_window: Width of the intervention window.
        attention_limitation_special: Keep only the leading special token visible.
        attention_limitation_special_all: Keep all special tokens visible.
        attention_limitation_question_only: Restrict intervention to question tokens.
        attention_limitation_question_right: Block instruction flow into sample tokens.
        attention_limitation_output: Apply intervention to generation positions.
        attention_limitation_input_output: Isolate sample-to-output flow.
        attention_limitation_instruction_output: Isolate instruction-to-output flow.
    """
    task = normalize_task_name(task)

    if attention_limitation_layer_from != -1 and attention_limitation_layer_to != -1:
        print("attention_limitation_layer_from and attention_limitation_layer_to not valid")
        return

    if sum([
        attention_limitation_special, attention_limitation_special_all, attention_limitation_question_only,
        attention_limitation_question_right, attention_limitation_input_output,
        attention_limitation_instruction_output
    ]) > 1:
        print("more than one attention limitation method selected - not valid")
        return

    seed_all(seed)

    if chat_template_model_name is None:
        chat_template_model_name = model_name

    cheat = False
    disturb = False
    add_question_mark = False
    bypass = False
    bypass_random = False
    bypass_wrong = False
    if "_question_mark" in questions:
        add_question_mark = True
        normalized_questions = questions.replace("_question_mark", "")
    elif "_cheat" in questions:
        cheat = True
        normalized_questions = questions.replace("_cheat", "")
    elif "_disturb" in questions:
        disturb = True
        normalized_questions = questions.replace("_disturb", "")
    elif questions == "bypass":
        bypass = True
        normalized_questions = questions
        if upfront:
            normalized_questions = '%s' % normalized_questions
            questions = f"upfront_{questions}"

        if zero:
            normalized_questions = '%s' % normalized_questions
            questions = f"zero_{questions}"

    elif questions == "bypass_random":
        bypass_random = True
        normalized_questions = questions
    elif questions == "bypass_wrong":
        bypass_wrong = True
        normalized_questions = questions
    elif upfront:
        normalized_questions = '%s' % questions
        questions = f"upfront_{questions}"
    elif zero:
        normalized_questions = '%s' % questions
        questions = f"zero_{questions}"
    else:
        normalized_questions = questions

    template_indices = [int(index) for index in template_index.split(",")]
    output_question_name = f"{questions}_{k}" if k > 0 else questions

    for parsed_template_index in template_indices:

        encoding_folder_structure = get_encoding_path(encoding_folder, task, model_name, chat_template_model_name,
                                                      model_precision, parsed_template_index, 1,
                                                      skip_layers=skip_layers,
                                                      attention_limitation_layer_to=attention_limitation_layer_to,
                                                      attention_limitation_layer_from = attention_limitation_layer_from,
                                                      attention_limitation_layer_window = attention_limitation_layer_window,
                                                      attention_limitation_special=attention_limitation_special,
                                                      attention_limitation_special_all=attention_limitation_special_all,
                                                      attention_limitation_question_only=attention_limitation_question_only,
                                                      attention_limitation_question_right=attention_limitation_question_right,
                                                      attention_limitation_output=attention_limitation_output,
                                                      attention_limitation_input_output=attention_limitation_input_output,
                                                      attention_limitation_instruction_output=attention_limitation_instruction_output,
                                                      )

        os.makedirs(encoding_folder_structure, exist_ok=True)

        if k == 0:
            if os.path.exists(
                    f"{encoding_folder_structure}/layer-0/generation_0_{questions}.feather") and not force_encoding:
                print("encodings already exists", encoding_folder_structure)
                continue
            if os.path.exists(
                    f"{encoding_folder_structure}/layer-0/generation_scored_0_{questions}.feather") and not force_encoding:
                print("encodings already exists", encoding_folder_structure)
                continue
        else:
            if os.path.exists(
                    f"{encoding_folder_structure}/layer-0/generation_0_{questions}_{k}.feather") and not force_encoding:
                print("encodings already exists", encoding_folder_structure)
                continue
            if os.path.exists(
                    f"{encoding_folder_structure}/layer-0/generation_scored_0_{questions}_{k}.feather") and not force_encoding:
                print("encodings already exists", encoding_folder_structure)
                continue

        model, tokenizer, chat_template_tokenizer = load_model_tokenizer(model_name, chat_template_model_name,
                                                                         model_architecture, model_precision, device,
                                                                         skip_layers=skip_layers)

        with open(TASK_TYPES_PATH) as task_types_file:
            task_types = json.load(task_types_file)
        if normalized_questions in ["zero", "zero_token", "random_prompt", "bypass", "bypass_random", "bypass_wrong"]:
            samples = pandas.read_json(resolve_task_path(task, "original"), lines=True, orient="records")
        else:
            samples = pandas.read_json(resolve_task_path(task, normalized_questions), lines=True, orient="records")

        if add_question_mark:
            samples["question"] = samples["question"].apply(lambda question: f"{question}?")

        if cheat:
            samples["question"] = samples.apply(lambda row: insert_cheat(task, row), axis=1)

        if disturb:
            samples["question"] = samples.apply(lambda row: insert_distribution(task, row), axis=1)

        if bypass:
            samples["question"] = samples.apply(lambda row: replace_repeat(row, task), axis=1)

        if bypass_random:
            samples["question"] = samples.apply(lambda row: replace_random(row), axis=1)

        if bypass_wrong:
            samples = samples.apply(lambda row: replace_wrong(row, task), axis=1)

        if device == "cpu":
            samples = samples.head(10)

        samples = samples[samples.notnull().all(1)]

        if k == 0:
            demonstration_samples = samples.copy()
        else:
            if normalized_questions in ["zero", "zero_token", "random_prompt", "bypass", "bypass_random",
                                        "bypass_wrong", "original", "original_disturb"]:
                demonstration_samples = samples.copy()
            else:
                demonstration_samples = pandas.read_json(
                    resolve_task_path(task, normalized_questions),
                    lines=True,
                    orient="records"
                )

        encoder = QuestionAnswerEncoder()

        if "zero" == normalized_questions:
            samples["instructions"] = samples.apply(lambda row: encoder.get_zero_instructions(
                task_types=task_types, entry=row, template_index=parsed_template_index,
                k=k, demonstrations=demonstration_samples, upfront=upfront, zero=zero
            ), axis=1)
        elif "zero_token" == normalized_questions:
            samples["instructions"] = samples.apply(lambda row: encoder.get_zero_token_instructions(
                task_types=task_types, entry=row, template_index=parsed_template_index,
                k=k, demonstrations=demonstration_samples, upfront=upfront, zero=zero
            ), axis=1)
        elif "random_prompt" in normalized_questions:
            samples["instructions"] = samples.apply(lambda row: encoder.get_randomized_instructions(
                task_types=task_types, entry=row, template_index=parsed_template_index,
                k=k, demonstrations=demonstration_samples, upfront=upfront, zero=zero
            ), axis=1)
        else:
            samples["instructions"] = samples.apply(lambda row: encoder.get_instructions(
                task_types=task_types, entry=row, template_index=parsed_template_index,
                k=k, demonstrations=demonstration_samples, upfront=upfront, zero=zero
            ), axis=1)

        def apply_chat_template(instruction):
            if chat_template_tokenizer.chat_template:
                if "system" not in chat_template_tokenizer.chat_template or "System role not supported" in chat_template_tokenizer.chat_template:
                    system_prompt = instruction[0]["content"]
                    first_query = instruction[1]['content']
                    instruction[1] = {
                        "role": "user",
                        "content": f"{system_prompt}\n\n{first_query}"
                    }

                    instruction = instruction[1:]

                return chat_template_tokenizer.apply_chat_template(conversation=instruction, tokenize=False,
                                                                   add_generation_prompt=True)
            else:
                return "\n".join([ele["content"] for ele in instruction]).strip()

        samples["compiled_instruction_text"] = samples["instructions"].apply(
            lambda instruction: apply_chat_template(instruction=instruction)
        )


        distinct_samples = samples.drop_duplicates("instance_id")

        dataset = Dataset.from_pandas(distinct_samples[["element_id", "instance_id", "sub_task", "question", "answer",
                                                        "compiled_instruction_text"]])

        input_encodings = []

        k_state_encodings = []
        q_state_encodings = []
        v_state_encodings = []
        o_state_encodings = []

        #attentions_encodings = []

        logit_scores = []

        instruction_encodings = []
        generated_encodings = []

        pbar = tqdm(dataset.iter(batch_size=encoding_batch_size))

        for batch in pbar:
            batch_frame = pandas.DataFrame(batch)

            encoded_batch = tokenizer(
                batch["compiled_instruction_text"], padding=True, truncation=True,
                return_tensors='pt', return_offsets_mapping=True, return_special_tokens_mask=True,
            ).to(device)

            relevant_samples = samples[samples["instance_id"].isin(batch["instance_id"])]

            all_token_span_indices = encoder.get_span_indices(encoded_batch, batch_frame, relevant_samples, tokenizer)

            if not zero:
                all_token_question_indices = encoder.get_question_indices(encoded_batch, batch_frame, relevant_samples, tokenizer)

            first_special_token = [ele.sum() for ele in encoded_batch["special_tokens_mask"]]

            batch_min_indices = [numpy.min(token_span_indices) for token_span_indices in all_token_span_indices]
            batch_max_indices = [numpy.max(token_span_indices) for token_span_indices in all_token_span_indices]

            all_special_tokens = [
                [
                    input_id_index
                    for input_id_index, input_id in enumerate(ele)
                    if input_id in tokenizer.all_special_ids and input_id_index < batch_min_indices[ele_index]
                ]
                for ele_index, ele in enumerate(encoded_batch["input_ids"])
            ]
            #last_non_special_token = [
            #    for token_id in encoded_batch["input_ids"]
            #]

            model_encoded_batch = {
                "input_ids": encoded_batch["input_ids"],
                "attention_mask": encoded_batch["attention_mask"],
            }

            decoded_batch = tokenizer.batch_decode(encoded_batch["input_ids"], skip_special_tokens=True)

            generation_config = GenerationConfig(**{
                # 'temperature': 0,
                # 'do_sample': False,
                # 'num_return_sequences': 1,
                # 'num_beams': 1,
                'max_new_tokens': max_new_tokens,
                # 'top_p': 1,
                #                'top_k': 0,
                'eos_token_id': tokenizer.eos_token_id,
                'pad_token': tokenizer.eos_token_id,
                'return_dict_in_generate': True,
                'output_hidden_states': True,
                #'output_attentions': True,
                'output_scores': True,
            })

            attention_cache = defaultdict(list)

            def save_span_vector(name, all_token_span_indices):
                """Capture mean span activations from projection outputs."""
                def hook(model, input, output):
                    if name not in attention_cache:
                        entry = torch.stack([
                            torch.concat([
                                output[i][span_indices].mean(dim=0) for span_indices in token_span_indices
                            ], dim=0) for i, token_span_indices in enumerate(all_token_span_indices)
                        ])
                        attention_cache[name].append(entry)

                return hook

            def save_generation_vector(name):
                """Capture projection outputs for generated tokens only."""
                def hook(model, input, output):
                    if output.shape[1] == 1:
                        attention_cache[name].append(output.detach().cpu())

                return hook
            def forward_wrap(original_forward, layer_index):
                """Wrap one attention module to apply masking interventions."""

                if "forward_wrap" in str(original_forward):
                    return original_forward
                def wrap(*args, **kwargs):

                    if attention_limitation_layer_to == -1 and attention_limitation_layer_from == -1:
                        return original_forward(*args, **kwargs)

                    elif layer_index > attention_limitation_layer_to and attention_limitation_layer_from == -1:
                        return original_forward(*args, **kwargs)

                    elif layer_index <= attention_limitation_layer_from and attention_limitation_layer_to == -1:
                        return original_forward(*args, **kwargs)

                    elif attention_limitation_layer_window != -1 and layer_index > attention_limitation_layer_from + attention_limitation_layer_window:
                        return original_forward(*args, **kwargs)

                    if kwargs["attention_mask"] is None:
                        batch_length = encoded_batch["input_ids"].shape[0]
                        length_1 = kwargs["hidden_states"].shape[1]
                        length_2 = kwargs["position_ids"].max() + 1

                        attn_mask = torch.ones((batch_length, 1, length_1, length_2)).to(encoded_batch["input_ids"].device)
                        attn_mask = attn_mask == 1
                    else:
                        attn_mask = kwargs["attention_mask"].detach().clone()

                    num_cols = attn_mask.shape[-1]
                    num_rows = attn_mask.shape[-2]

                    if model_precision == "half":
                        disable_value = -1e+4
                    else:
                        disable_value = -3.4028e+38

                    if attn_mask.max() != 0:
                        disable_value = False

                    if num_rows == 1:
                        ## generation step
                        if attention_limitation_output:
                            col_indices = torch.arange(num_cols, device=attn_mask.device)

                            for batch_idx in range(attn_mask.shape[0]):
                                disable_mask = col_indices > -1

                                if attention_limitation_question_only or attention_limitation_question_right:
                                    other_indices = [
                                        ele for ele in range(0, num_cols)
                                        if ele not in all_token_question_indices[batch_idx]
                                    ]
                                    disable_mask[other_indices] = False
                                elif attention_limitation_special_all:
                                    disable_mask[all_special_tokens[batch_idx]] = False
                                elif attention_limitation_special:
                                    disable_mask[first_special_token[0]] = False

                                attn_mask[batch_idx, 0, 0, disable_mask] = disable_value

                            kwargs["attention_mask"] = attn_mask

                        elif attention_limitation_input_output:
                            col_indices = torch.arange(num_cols, device=attn_mask.device)

                            for batch_idx in range(attn_mask.shape[0]):
                                disable_mask = col_indices < -1
                                relevant_tokens = list(itertools.chain.from_iterable(all_token_span_indices[batch_idx]))
                                disable_mask[min(relevant_tokens):] = True
                                attn_mask[batch_idx, 0, 0, disable_mask] = disable_value

                            kwargs["attention_mask"] = attn_mask

                        elif attention_limitation_instruction_output:
                            col_indices = torch.arange(num_cols, device=attn_mask.device)

                            for batch_idx in range(attn_mask.shape[0]):
                                disable_mask = col_indices < -1
                                other_indices = [
                                    ele for ele in range(0, num_cols)
                                    if ele in all_token_question_indices[batch_idx]
                                ]
                                disable_mask[other_indices] = True
                                attn_mask[batch_idx, 0, 0, disable_mask] = disable_value

                            kwargs["attention_mask"] = attn_mask

                        return original_forward(*args, **kwargs)


                    col_indices = torch.arange(num_cols, device=attn_mask.device)

                    for batch_idx, token_question_indices in enumerate(all_token_question_indices):
                        # These masks selectively block instruction-to-sample or instruction-to-output flow.

                        last_question_token_index = max(token_question_indices)
                        first_input_token_index = last_question_token_index + 1

                        disable_mask = (col_indices <= last_question_token_index)

                        if attention_limitation_question_only or attention_limitation_question_right:
                            other_indices = [ele for ele in range(0, min(all_token_question_indices[batch_idx]))]
                            disable_mask[other_indices] = False
                        elif attention_limitation_special_all:
                            disable_mask[all_special_tokens[batch_idx]] = False
                        elif attention_limitation_special:
                            disable_mask[first_special_token[0]] = False


                        if attention_limitation_question_right:
                            attn_mask[batch_idx, 0, first_input_token_index:-1, disable_mask] = disable_value

                            if attention_limitation_output:
                                attn_mask[batch_idx, 0, -1, disable_mask] = disable_value
                        elif attention_limitation_input_output:
                            relevant_tokens = list(itertools.chain.from_iterable(all_token_span_indices[batch_idx]))
                            disable_mask = (col_indices > min(relevant_tokens))
                            attn_mask[batch_idx, 0, -1, disable_mask] = disable_value
                        elif attention_limitation_instruction_output:
                            disable_mask =  (col_indices > min(all_token_question_indices[batch_idx])) & (col_indices <= max(all_token_question_indices[batch_idx]))
                            attn_mask[batch_idx, 0, -1, disable_mask] = disable_value


                    kwargs["attention_mask"] = attn_mask
                    return original_forward(*args, **kwargs)

                return wrap

            try:
                layers = model.model.layers
            except:
                layers = model.language_model.layers

            for i, layer in enumerate(layers):

                layer.self_attn.k_proj._forward_hooks = OrderedDict()
                layer.self_attn.q_proj._forward_hooks = OrderedDict()
                layer.self_attn.v_proj._forward_hooks = OrderedDict()
                layer.self_attn.o_proj._forward_hooks = OrderedDict()


                layer.self_attn.k_proj.register_forward_hook(save_span_vector(f"k_layer-{i + 1}", all_token_span_indices))
                layer.self_attn.q_proj.register_forward_hook(save_generation_vector(f"q_layer-{i + 1}"))
                layer.self_attn.v_proj.register_forward_hook(save_span_vector(f"v_layer-{i + 1}", all_token_span_indices))
                layer.self_attn.o_proj.register_forward_hook(save_generation_vector(f"o_layer-{i + 1}"))

                layer.self_attn.forward = forward_wrap(layer.self_attn.forward, layer_index=i)

            with torch.no_grad():
                generation = model.generate(
                    **model_encoded_batch,
                    generation_config=generation_config
                    )

            final_texts = tokenizer.batch_decode(generation.sequences, skip_special_tokens=True)

            generated_texts = [
                final_text.replace(decoded_input_text, "")
                for final_text, decoded_input_text in zip(final_texts, decoded_batch)
            ]

            #print("\n".join(generated_texts))
            # Representations are aggregated separately for sample-token spans and generated tokens.
            input_hidden_states = extract_input_hidden_state(generation, 1)

            output_hidden_states = extract_output_hidden_state(generation)

            input_encoding, instruction_encoding, k_state_encoding, q_state_encoding, v_state_encoding, o_state_encoding = encoder.get_input_instruction_hidden_state(
                input_hidden_states, attention_cache, encoded_batch, batch_frame, relevant_samples, tokenizer
            )

            #attention_elements = encoder.get_attention(
            #    generation.attentions, encoded_batch, batch_frame, relevant_samples, tokenizer
            #)

            score_elements = encoder.get_scores(
                generation.scores, encoded_batch, batch_frame, relevant_samples, tokenizer
            )


            #attentions_encodings.extend(attention_elements)
            logit_scores.extend(score_elements)

            k_state_encodings.extend(k_state_encoding)
            q_state_encodings.extend(q_state_encoding)
            v_state_encodings.extend(v_state_encoding)
            o_state_encodings.extend(o_state_encoding)

            input_encodings.extend(input_encoding)
            instruction_encodings.extend(instruction_encoding)

            generated_encodings.extend(
                encoder.get_generated_hidden_state(
                    output_hidden_states, encoded_batch, batch_frame, generated_texts, 1,
                )
            )

        k_state_frame = pandas.DataFrame(k_state_encodings).reset_index(drop=True)
        q_state_frame = pandas.DataFrame(q_state_encodings).reset_index(drop=True)
        v_state_frame = pandas.DataFrame(v_state_encodings).reset_index(drop=True)
        o_state_frame = pandas.DataFrame(o_state_encodings).reset_index(drop=True)
        #attention_frame = pandas.DataFrame(attentions_encodings).reset_index(drop=True)
        logit_score_frame = pandas.DataFrame(logit_scores).reset_index(drop=True)
        input_encoding_frame = pandas.DataFrame(input_encodings).reset_index(drop=True)
        instruction_encoding_frame = pandas.DataFrame(instruction_encodings).reset_index(drop=True)
        generated_encoding_frame = pandas.DataFrame(generated_encodings).reset_index(drop=True)

        if behavior_only:
            del k_state_frame["inputs_encoded"]
            del q_state_frame["inputs_encoded"]
            del v_state_frame["inputs_encoded"]
            del o_state_frame["inputs_encoded"]
            del input_encoding_frame["inputs_encoded"]
            del instruction_encoding_frame["inputs_encoded"]
            del generated_encoding_frame["inputs_encoded"]

        for frame in [input_encoding_frame, instruction_encoding_frame, generated_encoding_frame, k_state_frame,
                      q_state_frame, v_state_frame, o_state_frame, logit_score_frame]:
            if "instructions" in frame:
                del frame["instructions"]

            for col in frame.columns:
                if "level" in col:
                    del frame[col]

        k_state_frame["spans"] = k_state_frame["spans"].apply(
            lambda spans: "_".join([f"{span[0]},{span[1]},{span[2]}" for span in spans]))
        q_state_frame["spans"] = q_state_frame["spans"].apply(
            lambda spans: "_".join([f"{span[0]},{span[1]},{span[2]}" for span in spans]))
        v_state_frame["spans"] = v_state_frame["spans"].apply(
            lambda spans: "_".join([f"{span[0]},{span[1]},{span[2]}" for span in spans]))
        o_state_frame["spans"] = o_state_frame["spans"].apply(
            lambda spans: "_".join([f"{span[0]},{span[1]},{span[2]}" for span in spans]))
        #attention_frame["spans"] = attention_frame["spans"].apply(
        #    lambda spans: "_".join([f"{span[0]},{span[1]},{span[2]}" for span in spans]))
        logit_score_frame["spans"] = logit_score_frame["spans"].apply(
            lambda spans: "_".join([f"{span[0]},{span[1]},{span[2]}" for span in spans]))
        input_encoding_frame["spans"] = input_encoding_frame["spans"].apply(
            lambda spans: "_".join([f"{span[0]},{span[1]},{span[2]}" for span in spans]))
        instruction_encoding_frame["spans"] = instruction_encoding_frame["spans"].apply(
            lambda spans: "_".join([f"{span[0]},{span[1]},{span[2]}" for span in spans]))

        #for (sub_task, layer), layer_frame in attention_frame.groupby(["sub_task", "layer"]):
        #    sub_task_folder = encoding_folder_structure.replace(task, f'{task}-{sub_task}')
        #    os.system(f"mkdir -p {sub_task_folder}/layer-{layer}")
        #    relevant_columns = [col for col in layer_frame.columns if "__" not in col]
        #    layer_frame[relevant_columns].reset_index().to_feather(
        #        f"{sub_task_folder}/layer-{layer}/attention_{questions}.feather")

        #for sub_task, layer_frame in logit_score_frame.groupby("sub_task"):
        #    sub_task_folder = encoding_folder_structure.replace(task, f'{task}-{sub_task}')
        #    os.system(f"mkdir -p {sub_task_folder}/layer-0")
        #    relevant_columns = [col for col in layer_frame.columns if "__" not in col]
        #    layer_frame[relevant_columns].reset_index().to_feather(
        #        f"{sub_task_folder}/layer-0/scores_{questions}.feather")

        for (sub_task, layer), layer_frame in input_encoding_frame.groupby(["sub_task", "layer"]):
            sub_task_folder = encoding_folder_structure.replace(task, f'{task}-{sub_task}')
            os.makedirs(f"{sub_task_folder}/layer-{layer}", exist_ok=True)
            relevant_columns = [col for col in layer_frame.columns if "__" not in col]
            layer_frame[relevant_columns].reset_index().to_feather(
                f"{sub_task_folder}/layer-{layer}/input_{output_question_name}.feather")

        for (sub_task, layer), layer_frame in q_state_frame.groupby(["sub_task", "layer"]):
            sub_task_folder = encoding_folder_structure.replace(task, f'{task}-{sub_task}')
            os.makedirs(f"{sub_task_folder}/layer-{layer}", exist_ok=True)
            relevant_columns = [col for col in layer_frame.columns if "__" not in col]
            layer_frame[relevant_columns].reset_index().to_feather(
                f"{sub_task_folder}/layer-{layer}/q_state_{output_question_name}.feather")

        for (sub_task, layer), layer_frame in v_state_frame.groupby(["sub_task", "layer"]):
            sub_task_folder = encoding_folder_structure.replace(task, f'{task}-{sub_task}')
            os.makedirs(f"{sub_task_folder}/layer-{layer}", exist_ok=True)
            relevant_columns = [col for col in layer_frame.columns if "__" not in col]
            layer_frame[relevant_columns].reset_index().to_feather(
                f"{sub_task_folder}/layer-{layer}/v_state_{output_question_name}.feather")

        for (sub_task, layer), layer_frame in o_state_frame.groupby(["sub_task", "layer"]):
            sub_task_folder = encoding_folder_structure.replace(task, f'{task}-{sub_task}')
            os.makedirs(f"{sub_task_folder}/layer-{layer}", exist_ok=True)
            relevant_columns = [col for col in layer_frame.columns if "__" not in col]
            layer_frame[relevant_columns].reset_index().to_feather(
                f"{sub_task_folder}/layer-{layer}/o_state_{output_question_name}.feather")

        for (sub_task, layer), layer_frame in k_state_frame.groupby(["sub_task", "layer"]):
            sub_task_folder = encoding_folder_structure.replace(task, f'{task}-{sub_task}')
            os.makedirs(f"{sub_task_folder}/layer-{layer}", exist_ok=True)
            relevant_columns = [col for col in layer_frame.columns if "__" not in col]
            layer_frame[relevant_columns].reset_index().to_feather(
                f"{sub_task_folder}/layer-{layer}/k_state_{output_question_name}.feather")

        for (sub_task, layer), layer_frame in instruction_encoding_frame.groupby(["sub_task", "layer"]):
            sub_task_folder = encoding_folder_structure.replace(task, f'{task}-{sub_task}')
            os.makedirs(f"{sub_task_folder}/layer-{layer}", exist_ok=True)
            relevant_columns = [col for col in layer_frame.columns if "__" not in col]
            layer_frame[relevant_columns].reset_index().to_feather(
                f"{sub_task_folder}/layer-{layer}/instruction_{output_question_name}.feather")

        for (generation_id, layer), layer_frame in generated_encoding_frame.groupby(["generation_id", "layer"]):
            generation_id = start_generation_id + generation_id
            layer_frame["generation_id"] = generation_id
            os.makedirs(f"{encoding_folder_structure}/layer-{layer}", exist_ok=True)
            relevant_columns = [col for col in layer_frame.columns if "__" not in col]
            layer_frame[relevant_columns].reset_index().to_feather(
                f"{encoding_folder_structure}/layer-{layer}/generation_{generation_id}_{output_question_name}.feather")


if __name__ == "__main__":
    main()
