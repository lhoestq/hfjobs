# hfjobs

Hugging Face Jobs

## Installation

```
pip install hfjobs
```

## Available commands

```
usage: hfjobs <command> [<args>]

positional arguments:
  {inspect,logs,ps,run}
                        hfjobs command helpers
    inspect             Display detailed information on one or more Jobs
    logs                Fetch the logs of a Job
    ps                  List Jobs
    run                 Run a Job

options:
  -h, --help            show this help message and exit
```

## Run jobs

### Usage

```
usage: hfjobs <command> [<args>] run [-h] [-e ENV] [--env-file ENV_FILE] [--flavor FLAVOR] [-d] [--token TOKEN] dockerImage ...

positional arguments:
  dockerImage          The Docker image to use.
  command              The command to run.

options:
  -h, --help            show this help message and exit
  -e ENV, --env ENV     Set environment variables.
  -s SECRET, --secret SECRET
                        Set secret environment variables.
  --env-file ENV_FILE   Read in a file of environment variables.
  --secret-env-file SECRET_ENV_FILE
                        Read in a file of secret environment variables.
  --flavor FLAVOR       Flavor for the hardware, as in HF Spaces.
  --timeout TIMEOUT     Max duration: int/float with s (seconds, default), m (minutes), h (hours) or d (days).
  -d, --detach          Run the Job in the background and print the Job ID.
  --token TOKEN         A User Access Token generated from https://huggingface.co/settings/tokens
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
│    varchar    │
├───────────────┤
│ hello world   │
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
