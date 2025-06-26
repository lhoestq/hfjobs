# hfjobs Quickstart Guide

This quickstart will walk you through using `hfjobs` to run compute jobs on Hugging Face's infrastructure. By the end, you'll be running GPU workloads with simple commands.

## Installation & Setup

### Requirements

- Python 3.9 or higher
- A Hugging Face account (currently `hfjobs` is only available for HF staff)

### Install hfjobs

Install using pip:

```bash
pip install hfjobs
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install hfjobs
```

You can also run hfjobs directly without installation using `uv run`:

```bash
uv run hfjobs --help
```

### Verify Installation

Check that hfjobs is installed correctly:

```bash
hfjobs --help
```

You should see the help output with available commands and options.

## Authentication

hfjobs needs your Hugging Face token to submit jobs. The easiest way is to use the Hugging Face CLI:

```bash
huggingface-cli login
```

Follow the prompts to enter your token. hfjobs will automatically use your saved credentials.

To verify authentication is working:

```bash
hfjobs ps
```

This should show your jobs (or an empty list if you haven't run any yet).

## Your First Job

Let's run a simple Python command on Hugging Face's infrastructure:

```bash
hfjobs run python:3.12 python -c "print('Hello from the cloud!')"
```

This command:

- Uses the `python:3.12` Docker image
- Runs a Python one-liner that prints a message
- Executes on Hugging Face's infrastructure

You'll see output like:

```
Job started with ID: abc123xyz
View at: https://huggingface.co/jobs/username/abc123xyz
Hello from the cloud!
```

### Understanding the Output

- **Job ID**: Unique identifier for your job
- **Web URL**: Monitor your job in the browser
- **Logs**: Streamed in real-time to your terminal

### Watch Logs Stream in Real-Time

Let's run a longer job to see how logs are streamed:

```bash
hfjobs run python:3.12 python -c "
import time
print('Starting job...')
for i in range(5):
    print(f'Processing step {i+1}/5')
    time.sleep(2)
print('Job complete!')
"
```

You'll see each print statement appear as the job runs, giving you real-time feedback on your job's progress.

### Run in Detached Mode

For long-running jobs, you might not want to wait for output:

```bash
hfjobs run -d python:3.12 python -c "import time; time.sleep(300); print('Done!')"
```

This returns immediately with just the job ID. You can check on it later with:

```bash
hfjobs logs <job_id>
```

## Running Commands and Code

In the previous example, we passed Python code directly as a string. But hfjobs can run any command or program available in your container. Let's explore the different approaches.

### Understanding the Execution Model

When you run a job with hfjobs, your commands execute inside a container on Hugging Face's infrastructure. Since your local files aren't directly accessible in the container, you need strategies for running your programs.

### Execution Approaches

#### 1. Direct Commands

Run any command available in the container:

```bash
# Python code
hfjobs run python:3.12 python -c "print('Hello')"

# Shell commands
hfjobs run ubuntu:22.04 echo "Hello from Ubuntu"

# Data tools
hfjobs run hf.co/spaces/lhoestq/duckdb duckdb -c "SELECT 'Hello SQL'"
```

**When to use**: Quick tests, simple commands, one-liners

**Limitations**: Complex commands get unwieldy

#### 2. Download and Run

Fetch programs or scripts from URLs and execute them:

```bash
# Python script
hfjobs run python:3.12 /bin/bash -c \
  "wget https://example.com/script.py && python script.py"
```

**When to use**: Running existing code hosted online

**Limitations**: Dependencies must be handled separately

#### 3. UV Scripts

UV scripts include dependencies inline, making them perfect for hfjobs:

```bash
# Run our hello_world_uv.py example that uses cowsay
hfjobs run ghcr.io/astral-sh/uv:latest /bin/bash -c "
   uv run https://raw.githubusercontent.com/davanstrien/hfjobs/main/docs/examples/hello_world_uv.py 'Hello from the cloud!'"
```

The script includes its dependencies at the top:

```python
# /// script
# dependencies = [
#     "cowsay",
# ]
# ///
```

**When to use**: Scripts with dependencies, reproducible environments
**Benefits**: Dependencies handled automatically, no complex Docker builds

> See [`examples/hello_world_uv.py`](./examples/hello_world_uv.py) for the full script.

#### 4. Hugging Face Spaces

Use a Space as a container for complex projects with multiple files:

```bash
# Run a training script from a Space containing multiple modules
hfjobs run hf.co/spaces/username/my-training-space python train.py \
  --model bert-base --epochs 10

# The Space can contain:
# - train.py (main script)
# - model.py, data.py (supporting modules)
# - config.yaml (configuration files)
# - requirements.txt (dependencies)
```

**When to use**: Complex projects with multiple files, team collaboration
**Benefits**: Full project structure, version control, easy sharing

### Which Approach Should You Use?

- **Quick test or one-liner?** → Direct commands
- **Single script with dependencies?** → UV scripts
- **Complex project with multiple files?** → HF Space
- **Existing script online?** → Download and run

Each approach has its place. Start simple with direct commands, then move to UV scripts or Spaces as your needs grow.

## Choosing Your Container Image

The container image you choose determines what software is available to your code. Images are pre-built environments tailored for different workloads and frameworks.

### Quick Guide

Match your image to your task:

```bash
# Basic Python work → Python image
hfjobs run python:3.12 python -c "print('Hello')"

# PyTorch code → PyTorch image
hfjobs run pytorch/pytorch:latest python -c "import torch; print(torch.__version__)"

# Using UV scripts without GPU → UV image
hfjobs run ghcr.io/astral-sh/uv:latest uv run script.py
```

### Common Images

- **python:3.12** - Clean Python environment
- **ubuntu:22.04** - Commonly used linux image
- **pytorch/pytorch** - PyTorch pre-installed
- **tensorflow/tensorflow** - TensorFlow pre-installed
- **ghcr.io/astral-sh/uv** - An image with uv set up for running UV scripts
- **huggingface/transformers-pytorch-gpu** - Hugging Face libraries ready to go

### GPU Considerations

For GPU workloads, use CUDA-enabled images:

```bash
# CPU version
hfjobs run pytorch/pytorch:latest python -c "..."

# GPU version (note the cuda tag)
hfjobs run --flavor t4-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel python -c "..."
```

## Working with Different Hardware

So far we've been using the default CPU hardware. The real power of hfjobs comes from accessing GPUs (or TPUs) with a simple flag.

### Run on GPU

Let's verify CUDA is available on a GPU instance:

```bash
hfjobs run --flavor t4-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name()}')"
```

Output:

```
CUDA available: True
GPU: NVIDIA T4
```

### Check GPU Memory

Let's see how much memory is available on a T4:

```bash
hfjobs run --flavor t4-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name()}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')
"
```

### Available Hardware Options

Some of the most common hardware flavors you can use with hfjobs:

| Flavor        | Hardware        | GPU Memory | Best For                             |
| ------------- | --------------- | ---------- | ------------------------------------ |
| `cpu-basic`   | CPU only        | N/A        | Light processing, debugging          |
| `cpu-upgrade` | High-memory CPU | N/A        | Data processing, CPU-intensive tasks |
| `t4-small`    | NVIDIA T4       | 16 GB      | Inference, small models              |
| `l4x1`        | NVIDIA L4       | 24 GB      | Modern inference, fine-tuning        |
| `a10g-small`  | NVIDIA A10G     | 24 GB      | Medium training jobs                 |
| `a10g-large`  | NVIDIA A10G     | 24 GB      | Larger batch sizes                   |
| `a100-large`  | NVIDIA A100     | 80 GB      | Large model training                 |

You can find the full list of available flavours in the [Hub Docs](https://huggingface.co/docs/hub/spaces-gpus#hardware-specs)

## Environment Variables and Secrets

Your jobs often need configuration values or credentials. hfjobs provides two ways to pass these securely.

### Environment Variables

Pass configuration values using the `-e` flag:

```bash
hfjobs run -e MODEL_NAME=bert-base-uncased -e BATCH_SIZE=32 \
  python:3.12 python -c "
import os
print(f'Model: {os.environ[\"MODEL_NAME\"]}')
print(f'Batch size: {os.environ[\"BATCH_SIZE\"]}')
"
```

### Secrets

For sensitive values like API keys, use the `--secret` flag:

```bash
# Pass a secret value to the job
hfjobs run --secret HF_TOKEN=hf_*** \
  python:3.12 python -c "
import os
token = os.environ.get('HF_TOKEN', 'not set')
print(f'Token available: {\"yes\" if token != \"not set\" else \"no\"}')
"
```

For local environment variables, you need to explicitly pass the value:

```bash
# Pass your local HF token to the job
hfjobs run --secret HF_TOKEN="${HF_TOKEN}" \
  python:3.12 python -c "
import os
# This will print 'yes' but the actual token value is masked in logs
print(f'Token available: {\"yes\" if os.environ.get('HF_TOKEN') else \"no\"}')
"
```

The key difference: environment variables are visible in logs, while secrets are masked for security.

## Managing Your Jobs

Once you've submitted jobs, you'll need to monitor and manage them. hfjobs provides commands similar to Docker for job management.

### List Your Jobs

See all your running jobs:

```bash
hfjobs ps
```

Output:

```
JOB ID      IMAGE         COMMAND                STATUS    CREATED
abc123xyz   python:3.12   python train.py        RUNNING   2 minutes ago
def456uvw   python:3.12   python inference.py    COMPLETED 1 hour ago
```

Filter by status:

```bash
# Show only running jobs
hfjobs ps --filter status=running

# Show all jobs including completed ones
hfjobs ps --all
```

### View Job Logs

Check the output of a specific job:

```bash
hfjobs logs abc123xyz
```

For long-running jobs, you might want to check logs periodically:

```bash
# Show logs with timestamps
hfjobs logs -t abc123xyz
```

### Cancel a Job

Stop a running job:

```bash
hfjobs cancel abc123xyz
```

This immediately terminates the job and frees up the resources.

## It's not just for Python!

While our examples focus on Python, hfjobs works with **any language or tool** available in Docker containers.

### Compile and Run Rust

```bash
hfjobs run rust:latest /bin/bash -c "
echo 'fn main() { println!(\"Hello from Rust! 🦀\"); }' > hello.rs &&
rustc hello.rs &&
./hello
"
```

### Run Node.js Applications

```bash
hfjobs run node:20 node -e "
console.log('Node.js version:', process.version);
console.log('Computing fibonacci(40)...');
const fib = (n) => n <= 1 ? n : fib(n-1) + fib(n-2);
console.log('Result:', fib(40));
"
```

### Use CLI Tools

Process data with standard Unix tools:

```bash
# Analyze a dataset with jq
hfjobs run ubuntu:22.04 /bin/bash -c "
curl -s https://api.github.com/repos/huggingface/transformers |
jq '{name: .name, stars: .stargazers_count, language: .language}'
"
```

The key point: if it runs in a container, it runs on hfjobs!

## Next Steps

Now that you've learned the basics of hfjobs, you can:

- **Explore more examples**: Check out the [hello_world_uv.py](./examples/hello_world_uv.py) script

Happy computing! 🚀
