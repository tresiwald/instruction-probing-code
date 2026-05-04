# Instructions Shape Language Production, Not Processing

Code and data for the paper *Instructions Shape Language Production, Not Processing*.

This project studies a narrow but important question: when instructions affect model behavior, where does that effect live? The repository is organized around a simple experimental pipeline:

1. encode hidden representations for instruction-conditioned task examples
2. probe those representations with a classifier-based probing backend

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

### 1. Encode Activations

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
- `encoding_batch_size` (optional): batch size for the `k=4` runs. Default: `2`.

Encoded activations are written to `encodings/`.

### 2. Probe Representations

The probing job reads those saved activations and runs linear probes through the bundled probing backend:

```bash
bash src/jobs/probe_base.bash \
  Qwen/Qwen2.5-7B-Instruct \
  blimp,olmpics,stereoset,tomi
```

Optional arguments:

- `precision`: defaults to `full`.
- `processing`: defaults to `parallel`.

Probe outputs are written to `results/`.

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
