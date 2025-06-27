# Using UV to Run Scripts with hfjobs

This guide explains how to use uv to run scripts with hfjobs.

## What is UV?

[UV](https://docs.astral.sh/uv) is a Python package manager that can run Python scripts. The simplest way to use UV with hfjobs is to run any Python script:

```bash
# Run a script from a URL
hfjobs run ghcr.io/astral-sh/uv:debian-slim uv run https://example.com/script.py
```

This works with any Python script - no special setup required!

On its own, this isn't very exciting; you can also run a Python script directly with Python! One of the features that makes UV more powerful is the ability to declare dependencies directly in your Python scripts, which allows you to run them without needing to install any dependencies manually.

### Install UV

See [the UV documentation](https://docs.astral.sh/uv/installation/) for up to date installation instructions.

### UV Scripts: Adding Dependencies

Let's look at a simple example of a Python script with dependencies. This script relies on the [`cowsay`](https://pypi.org/project/cowsay/) library to print a message:

```python
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "cowsay",
# ]
# ///
"""A simple UV script example for hfjobs.
This script demonstrates how UV scripts can specify their dependencies
inline, making them perfect for running with hfjobs.
"""

import cowsay
import sys


def main():
    message = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello from hfjobs!"
    cowsay.cow(message)


if __name__ == "__main__":
    main()
```

If we have the script saved as `hello_world_uv.py`, you can run it locally (assuming you have uv installed) like this:

```bash
uv run hello_world_uv.py "Hello from my CLI!"
```

We can also run uv scripts via a URL:

```bash
uv run https://raw.githubusercontent.com/davanstrien/hfjobs/refs/heads/quickstart-only/docs/examples/hello_world_uv.py "Hello from my CLI, I arrived from the internet via a URL!"
```

Now, to run it on Hugging Face infrastructure using hfjobs we would simply need to run instead:

<!-- TODO: Update these URLs to point to the main branch once the examples are merged and published -->

```bash
hfjobs run ghcr.io/astral-sh/uv:debian-slim uv run https://raw.githubusercontent.com/davanstrien/hfjobs/refs/heads/quickstart-only/docs/examples/hello_world_uv.py "Hello from hfjobs!"
```

This command runs your script on Hugging Face's infrastructure, automatically installing cowsay in an isolated environment. We'll explain how this works in detail later, but the key point is that you can run any Python script with dependencies on Hugging Face infrastructure using a single command!

### Why UV Scripts + hfjobs?

UV scripts solve a fundamental challenge when running code on remote infrastructure: dependency management. Instead of building Docker images or manually installing packages, UV scripts let you declare dependencies right in your Python file.

**Key benefits for hfjobs users:**

- **Zero setup**: Your script runs anywhere with just a URL - no Docker knowledge needed
- **Self-contained**: Dependencies travel with your code, ensuring reproducibility
- **Instant iteration**: Change dependencies without rebuilding containers
- **Perfect for sharing**: Send colleagues a single command that just works

**Ideal for ML workflows:**

UV scripts are particularly powerful for machine learning tasks. That `train.py` script you've been working on? Add a UV header with your dependencies, and it's ready to run on GPUs with hfjobs. When your script includes a CLI (using argparse or click), you get a flexible tool that can handle different datasets, models, and hyperparameters - we'll show examples of this pattern throughout the guide.

You can think of UV scripts as "portable cloud functions" - your Python script becomes a complete, runnable unit that hfjobs can execute on any hardware with one command.

## Getting Started

Let's create and run your first UV script on Hugging Face's infrastructure.

### 1. Create a UV Script

First, create a new UV script using the `uv init` command:

```bash
uv init --script process_data.py
```

This creates a template script:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

def main():
    print("Hello from UV!")

if __name__ == "__main__":
    main()
```

### 2. Add Dependencies

Add the packages your script needs:

```bash
# For data processing
uv add --script process_data.py pandas pyarrow requests

# For machine learning
uv add --script process_data.py torch transformers datasets
```

Your script header now includes the dependencies:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "pyarrow",
#     "requests",
#     "torch",
#     "transformers",
#     "datasets",
# ]
# ///
```

### 3. Test Locally

Make sure your script works:

```bash
uv run process_data.py
```

### 4. Upload to Hugging Face Hub

Create a dataset repository for your scripts:

```bash
# Create a dataset repo (only needed once)
huggingface-cli repo create my-uv-scripts --type dataset

# Upload your script
huggingface-cli upload my-uv-scripts process_data.py scripts/process_data.py --repo-type dataset
```

### 5. Run with hfjobs

Now run your script on HF infrastructure:

```bash
# CPU execution
hfjobs run ghcr.io/astral-sh/uv:debian-slim /bin/bash -c \
  "uv run https://huggingface.co/datasets/{your-username}/my-uv-scripts/raw/main/scripts/process_data.py"

# GPU execution
hfjobs run --flavor gpu-nvidia-small ghcr.io/astral-sh/uv:debian-slim /bin/bash -c \
  "uv run https://huggingface.co/datasets/{your-username}/my-uv-scripts/raw/main/scripts/process_data.py"
```

That's it! Your script is running on Hugging Face's infrastructure with all dependencies automatically installed.

## Key Concepts for Running UV Scripts

### Basic Command Pattern

The pattern for running UV scripts with hfjobs is:

```bash
hfjobs run <docker_image> /bin/bash -c "uv run <script_url> <args>"
```

The `/bin/bash -c` wrapper allows us to run shell commands (like setting environment variables) before executing the UV script.

For most cases, use the lightweight UV image:

- **`ghcr.io/astral-sh/uv:debian-slim`** - Fast startup, includes UV and Python

### Common Options

**Running on GPU:**

```bash
hfjobs run --flavor gpu-nvidia-small ghcr.io/astral-sh/uv:debian-slim /bin/bash -c \
  "uv run https://huggingface.co/datasets/{username}/my-scripts/raw/main/train.py"
```

**Passing secrets (like HF token):**

```bash
hfjobs run --secret HF_TOKEN=$HF_TOKEN ghcr.io/astral-sh/uv:debian-slim /bin/bash -c \
  "uv run https://huggingface.co/datasets/{username}/my-scripts/raw/main/upload.py"
```

**Setting environment variables:**

```bash
hfjobs run ghcr.io/astral-sh/uv:debian-slim /bin/bash -c \
  "export HOME=/tmp && uv run your_script.py"
```

For advanced topics like Docker image selection, environment setup, and system dependencies, see the [advanced guide](./uv_scripts_advanced.md).

## Example: Process a Hugging Face Dataset

Here's a complete example that downloads and analyzes a dataset:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "pandas",
# ]
# ///

import argparse
from datasets import load_dataset
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset name (e.g., 'imdb')")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    # Load dataset
    print(f"Loading {args.dataset}...")
    ds = load_dataset(args.dataset, split=f"train[:{args.max_samples}]")

    # Basic analysis
    df = pd.DataFrame(ds)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst example:")
    print(df.iloc[0].to_dict())

if __name__ == "__main__":
    main()
```

Run this example:

```bash
# Upload to HF Hub
huggingface-cli upload my-uv-scripts analyze.py scripts/analyze.py --repo-type dataset

# Run on CPU
hfjobs run ghcr.io/astral-sh/uv:debian-slim /bin/bash -c \
  "uv run https://huggingface.co/datasets/{username}/my-uv-scripts/raw/main/scripts/analyze.py imdb --max-samples 1000"
```

You should see output like:

```python
Loading imdb...
train-00000-of-00001.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21.0M/21.0M [00:00<00:00, 64.4MB/s]
...
Dataset shape: (100, 2)
Columns: ['text', 'label']
First example:
...
```

## Saving Your Results

When your script runs on Hugging Face infrastructure, any output to stdout is displayed in your terminal. Often though, you don't just want to print results; you want to save them to a file or upload them somewhere.

You can do this in a few ways:

### Option 1: Use existing push_to_hub functionality

The Transformers, TRL, datasets libraries (and many more!) can push results to the Hugging Face Hub directly using their built-in `push_to_hub` functionality. This is the recommended way to save models, datasets, and other artifacts. This means you can use the same code you would use locally to save your results, and it will work seamlessly on Hugging Face's infrastructure.

### Option 2: Upload results using the `huggingface-hub` library

Add the `huggingface-hub` library to your script and upload results directly:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
#     "pandas",
#     "huggingface-hub",
# ]
# ///

from huggingface_hub import HfApi
import os

# Your processing code here...

# Upload results
api = HfApi()
api.upload_file(
    path_or_fileobj="results.csv",
    path_in_repo="outputs/results.csv",
    repo_id="username/my-results",
    repo_type="dataset",
    token=os.environ.get("HF_TOKEN")
)
```

### Option 3: Use a directory to store results

You can also write results to a directory and then upload that directory as a dataset. For example if you were saving multiple checkpoints or filtered version of a dataset to a `output` directory you could use `upload_folder` to upload to the hub (or use `upload_large_folder` if you are uploading a large amount of data).

## Quick Reference

### Essential Commands

```bash
# Create UV script
uv init --script myscript.py

# Add dependencies
uv add --script myscript.py pandas torch

# Test locally
uv run myscript.py

# Upload to HF
huggingface-cli upload my-uv-scripts myscript.py scripts/myscript.py --repo-type dataset

# Run on CPU
hfjobs run ghcr.io/astral-sh/uv:debian-slim /bin/bash -c \
  "uv run https://huggingface.co/datasets/{username}/my-uv-scripts/raw/main/scripts/myscript.py"

# Run on GPU
hfjobs run --flavor gpu-nvidia-small ghcr.io/astral-sh/uv:debian-slim /bin/bash -c \
  "uv run https://huggingface.co/datasets/{username}/my-uv-scripts/raw/main/scripts/myscript.py"

# With secrets
hfjobs run --secret HF_TOKEN=$HF_TOKEN ghcr.io/astral-sh/uv:debian-slim /bin/bash -c \
  "uv run your_script.py"
```

### Getting Help

- **UV documentation**: https://docs.astral.sh/uv/
- **hfjobs documentation**: https://github.com/huggingface/hfjobs

## Next Steps

You now have everything you need to run UV scripts on Hugging Face's infrastructure! Try:

1. Modifying the example for your use case
2. Exploring GPU options for ML workloads
3. Building a collection of reusable scripts

Happy scripting! 🚀
