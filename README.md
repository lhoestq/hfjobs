# 🚀 hfjobs

Run compute jobs on Hugging Face infrastructure with a familiar Docker-like interface!

`hfjobs` is a command-line tool that lets you run anything on Hugging Face's infrastructure (including GPUs and TPUs!) with simple commands. Think `docker run`, but for running code on A100s.

```bash
# Directly run Python code
hfjobs run python:3.12 python -c "print('Hello from the cloud!')"

# Use GPUs without any setup
hfjobs run --flavor a10g-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  python -c "import torch; print(torch.cuda.get_device_name())"

# Run from Hugging Face Spaces
hfjobs run hf.co/spaces/lhoestq/duckdb duckdb -c "select 'hello world'"
```

## ✨ Key Features

- 🐳 **Docker-like CLI**: Familiar commands (`run`, `ps`, `logs`, `inspect`) to run and manage jobs
- 🔥 **Any Hardware**: From CPUs to A100 GPUs and TPU pods - switch with a simple flag
- 📦 **Run Anything**: Use Docker images, HF Spaces, or your custom containers
- 🔐 **Simple Auth**: Just use your HF token
- 📊 **Live Monitoring**: Stream logs in real-time, just like running locally
- 💰 **Pay-as-you-go**: Only pay for the seconds you use

## Prerequisites

- Python 3.9 or higher
- A Hugging Face account (currently in testing for HF staff)

## Installation

Install with pip:

```bash
pip install hfjobs
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install hfjobs
```

It's also possible to run `hfjobs` without installing it, using [uv run](https://docs.astral.sh/uv/):

```bash
uv run hfjobs --help
```

## Quick Start

### 1. Authenticate with the Hugging Face Hub

```bash
huggingface-cli login
```

or export your Hugging Face token as an environment variable:

```bash
export HF_TOKEN="your_token_here"
```

### 2. Run your first job

```bash
# Run a simple Python script
hfjobs run python:3.12 python -c "print('Hello from HF compute!')"
```

### 3. Check job status

```bash
# List your running jobs
hfjobs ps

# View logs from a specific job
hfjobs logs <job_id>
```

### 4. Run on GPU

You can also run jobs on GPUs or TPUs with the `--flavor` option. For example, to run a PyTorch job on an A10G GPU:

```bash
# Use an A10G GPU to check PyTorch CUDA
hfjobs run --flavor a10g-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  python -c "import torch; print(f"This code ran with the following GPU: {torch.cuda.get_device_name()}")"
```

Running this will show the following output!

```bash
This code ran with the following GPU: NVIDIA A10G
```

That's it! You're now running code on Hugging Face's infrastructure. For more detailed information, see the sections below.

## Common Use Cases

- **Model Training**: Fine-tune or train models on GPUs (T4, A10G, A100) without managing infrastructure
- **Synthetic Data Generation**: Generate large-scale datasets using LLMs on powerful hardware
- **Data Processing**: Process massive datasets with high-CPU configurations for parallel workloads
- **Batch Inference**: Run offline inference on thousands of samples using optimized GPU setups
- **Experiments & Benchmarks**: Run ML experiments on consistent hardware for reproducible results
- **Development & Debugging**: Test GPU code without local CUDA setup

## Available commands

```

usage: hfjobs <command> [<args>]

positional arguments:
{inspect,logs,ps,run,cancel}
hfjobs command helpers
inspect Display detailed information on one or more Jobs
logs Fetch the logs of a Job
ps List Jobs
run Run a Job
cancel Cancel a Job

options:
-h, --help show this help message and exit

```

## Run jobs

### Usage

```

usage: hfjobs <command> [<args>] run [-h] [-e ENV] [-s SECRET] [--env-file ENV_FILE] [--secret-env-file SECRET_ENV_FILE] [--flavor FLAVOR] [--timeout TIMEOUT] [-d] [--token TOKEN] dockerImage ...

positional arguments:
dockerImage The Docker image to use.
command The command to run.

options:
-h, --help show this help message and exit
-e ENV, --env ENV Set environment variables.
-s SECRET, --secret SECRET
Set secret environment variables.
--env-file ENV_FILE Read in a file of environment variables.
--secret-env-file SECRET_ENV_FILE
Read in a file of secret environment variables.
--flavor FLAVOR Flavor for the hardware, as in HF Spaces.
--timeout TIMEOUT Max duration: int/float with s (seconds, default), m (minutes), h (hours) or d (days).
-d, --detach Run the Job in the background and print the Job ID.
--token TOKEN A User Access Token generated from https://huggingface.co/settings/tokens

```

### Examples

```

$ hfjobs run ubuntu echo hello world
hello world

```

```

$ hfjobs run python:3.12 python -c "print(2+2)"
4

```

```

$ hfjobs run python:3.12 /bin/bash -c "cd /tmp && wget https://gist.githubusercontent.com/sergeyprokudin/e8e1eeb9263766cc43a05ab9190442e4/raw/3c34504fd646517aeb15903700f8e9c1f4d6d2e5/fibonacci.py && python fibonacci.py"
0
1
...
218922995834555169026

```

```

$ hfjobs run hf.co/spaces/lhoestq/duckdb duckdb -c "select 'hello world'"
┌───────────────┐
│ 'hello world' │
│ varchar │
├───────────────┤
│ hello world │
└───────────────┘

```

```

$ hfjobs run --flavor t4-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel python -c "import torch; print(torch.tensor([42]).to('cuda'))"
tensor([42], device='cuda:0')

```

## Hardware

Available `--flavor` options:

- CPU: `cpu-basic`, `cpu-upgrade`
- GPU: `t4-small`, `t4-medium`, `l4x1`, `l4x4`, `a10g-small`, `a10g-large`, `a10g-largex2`, `a10g-largex4`,`a100-large`
- TPU: `v5e-1x1`, `v5e-2x2`, `v5e-2x4`

(updated in 03/25 from Hugging Face [suggested_hardware docs](https://huggingface.co/docs/hub/en/spaces-config-reference))
