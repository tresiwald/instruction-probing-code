<p align="center">
  <img src="assets/logo.png" alt="Instructions Shape Language Production logo" width="140">
</p>

<h1 align="center">
  Instructions Shape <i><span style="color: #f08000;">Production</span></i> of Language,<br>
  not <del><i><span style="color: #00a0a8;">Processing</span></i></del>
</h1>

<p align="center">
  <a href="https://instruction-probing.github.io/">Project Page</a>
</p>

Code and data for the paper *Instructions Shape Language Production, Not Processing*.

This project studies a narrow but important question: when instructions affect model behavior, where does that effect live? The repository is organized around a simple experimental pipeline:

1. dump task representations for instruction-conditioned task examples
2. probe those dumped representations with a classifier-based probing backend

## Overview

- `tasks/`: released task datasets used in the paper (`blimp`, `olmpics`, `stereoset`, `tomi`)
- `defs/task_types.json`: prompt templates used during encoding
- `src/encode.py`: encoding entry point
- `src/probe.py`: probing entry point
- `src/jobs/encode_tasks.bash`: batch script for encoding runs
- `src/jobs/probe_base.bash`: batch script for probing runs
- `src/probing`: probing backend, included as a Git submodule

## Setup

Use Python 3.10. After cloning, initialize the probing submodule on its `probe_only` branch and install dependencies:

```bash
git submodule update --init --recursive
git -C src/probing checkout probe_only
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Optional: Redis Tracking

This repository can use Redis for experiment tracking, similar to the setup used in [aligned-probing](https://github.com/alignedprobing/aligned-probing).

Start a local Redis instance with Docker Compose:

```yaml
networks:
  experiments:
    name: experiments
    driver: bridge

services:
  redis:
    image: redis:latest
    ports:
      - "6379:6379"
    networks:
      - experiments
    volumes:
      - ../redis-data:/data

  redis-insight:
    image: redis/redisinsight:latest
    ports:
      - "5540:5540"
    networks:
      - experiments
```

Then export the tracking variables before running probes with `--logging redis`:

```bash
export REDIS_SERVER=127.0.0.1
export REDIS_PORT=6379
```

If you do not need experiment tracking, nothing else is required: the default job scripts use local logging.

## Run The Pipeline

### 1. Encode Tasks

The encoding job script runs the main prompt variants used in the paper for a comma-separated task list:

```bash
bash src/jobs/encode_tasks.bash \
  Qwen/Qwen2.5-7B-Instruct \
  blimp,olmpics,stereoset,tomi \
  0 \
  full
```

Arguments:

- `model_name`: Hugging Face model identifier.
- `tasks`: comma-separated task names.
- `device`: CUDA device string, for example `0` or `0,1`. Use `cpu` for CPU runs.
- `precision`: `full`, `half`, or `four_bit`.

Dumped task representations are written to `encodings/`.

### 2. Probe Tasks

The probing job reads those dumped task representations and runs linear probes through the bundled probing backend:

```bash
bash src/jobs/probe_base.bash \
  Qwen/Qwen2.5-7B-Instruct \
  blimp,olmpics,stereoset,tomi
```

Optional arguments:

- `precision`: defaults to `full`.
- `processing`: defaults to `parallel`.

Probe outputs are written to `results/`.

## Extract Results

This repository produces two kinds of outputs:

1. behavioral outputs from model generations
2. internal outputs from dumped hidden-state representations and probe runs

### Behavioral Outputs

Behavioral generations are stored as Feather files under `encodings/<task>/<model>/<chat_template_model>/<precision>/template-<id>/layer-*/generation_*_<question>.feather`.

For example:

```bash
encodings/blimp/Qwen_Qwen2.5-7B-Instruct/Qwen_Qwen2.5-7B-Instruct/full/template-0/layer-0/generation_0_original.feather
```

These files contain, among others, the following columns:

- `question`
- `answer`
- `generation_text`
- `layer`
- `generation_id`
- `inputs_encoded`

To extract behavioral predictions:

```python
import pandas as pd

df = pd.read_feather(
    "encodings/blimp/Qwen_Qwen2.5-7B-Instruct/Qwen_Qwen2.5-7B-Instruct/full/template-0/layer-0/generation_0_original.feather"
)

behavior = df[["question", "answer", "generation_text", "layer", "generation_id"]]
print(behavior.head())
```

### Internal Outputs

Internal task representations are stored as Feather files under subtask-specific folders such as `blimp-default`, `olmpics-default`, or intervention variants such as `blimp-default-sep-output-till-quest-right-0`.

The two retained internal files are:

- `sample_<question>.feather`: span-level sample representations
- `output_<question>.feather`: output projection representations

For example:

```bash
encodings/blimp-default/Qwen_Qwen2.5-7B-Instruct/Qwen_Qwen2.5-7B-Instruct/full/template-0/layer-0/sample_original.feather
```

These files contain task metadata together with the representation vector in `inputs_encoded`. Typical metadata columns include:

- `task_type`
- `sub_task`
- `context`
- `question`
- `answer`
- `label`
- `layer`
- `inputs_encoded`

To extract sample-side internal representations:

```python
import pandas as pd

df = pd.read_feather(
    "encodings/blimp-default/Qwen_Qwen2.5-7B-Instruct/Qwen_Qwen2.5-7B-Instruct/full/template-0/layer-0/sample_original.feather"
)

internal = df[["context", "question", "answer", "label", "layer", "inputs_encoded"]]
print(internal.head())
```

To extract output-side internal representations:

```python
import pandas as pd

df = pd.read_feather(
    "encodings/blimp-default/Qwen_Qwen2.5-7B-Instruct/Qwen_Qwen2.5-7B-Instruct/full/template-0/layer-1/output_original.feather"
)

output = df[["context", "question", "answer", "label", "layer", "inputs_encoded"]]
print(output.head())
```

### Probe Outputs

Probing results are stored under `results/` and include:

- `metrics.csv`: validation and test metrics
- `preds.csv`: per-example probe predictions
- `hparams.yaml`: probe configuration
- `*.ckpt`: saved probe checkpoints

For example:

```bash
results/lift-pair-blimp-default/Qwen_Qwen2.5-7B-Instruct/full/NONE/5000/0/0/done/metrics.csv
```

To extract summary metrics and probe predictions:

```python
import pandas as pd

metrics = pd.read_csv(
    "results/lift-pair-blimp-default/Qwen_Qwen2.5-7B-Instruct/full/NONE/5000/0/0/done/metrics.csv"
)
preds = pd.read_csv(
    "results/lift-pair-blimp-default/Qwen_Qwen2.5-7B-Instruct/full/NONE/5000/0/0/done/preds.csv"
)

print(metrics.tail(1))
print(preds.head())
```

In practice:

- use `generation_*.feather` when you want behavioral outputs
- use `sample_*.feather` and `output_*.feather` when you want internal representations
- use `metrics.csv` and `preds.csv` under `results/` when you want probe-level internal evaluation results

## EWOK

`ewok` is available upon request and is therefore not part of the default public release. If you receive access to it, place the file at `tasks/ewok.jsonl` and include `ewok` in the task list passed to the job scripts.

## Notes

- The repository expects local encoding and probing runs; the default job scripts use local logging rather than the original remote Redis bookkeeping.
- Redis-based experiment tracking is optional and can be enabled with `--logging redis` together with `REDIS_SERVER` and `REDIS_PORT`.
- The probing backend is kept as a Git submodule rather than copied directly into this repository.

## Citation

If you use this repository, cite the paper together with the probing benchmark:

```bibtex
@misc{waldis2025instructions,
  title        = {Instructions shape Production of Language, not Processing},
  author       = {Waldis, Andreas and Choshen, Leshem and Hou, Yufang and Perlitz, Yotam},
  year         = {2025},
  note         = {Manuscript accompanying this repository}
}

@article{waldis2024holmes,
  title        = {Holmes: A Benchmark to Assess the Linguistic Competence of Language Models},
  author       = {Waldis, Andreas and Perlitz, Yotam and Choshen, Leshem and Hou, Yufang and Gurevych, Iryna},
  journal      = {Transactions of the Association for Computational Linguistics},
  volume       = {12},
  pages        = {1616--1647},
  year         = {2024},
  doi          = {10.1162/tacl_a_00718}
}
```
