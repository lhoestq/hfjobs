# Using UV to Run Scripts with hfjobs

This guide explains how to use uv to run scripts with hfjobs.

## What is UV?

UV is a Python package manager that can run Python scripts directly. The simplest way to use UV with hfjobs is to run any Python script:

```bash
# Run a script from a URL
hfjobs run ghcr.io/astral-sh/uv:debian-slim uv run https://example.com/script.py
```

This works with any Python script - no special setup required!

On its own, this isn't very exciting, you can also run a python script directly with Python! One of the things that makes uv more powerful is the ability to declare dependencies directly in your Python scripts, which allows you to run them without needing to install anything manually.

### UV Scripts: Adding Dependencies

Let's look at a simple example of a Python script with dependencies. This script relies on the `cowsay` library to print a message:

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

Now, to run it on Hugging Face infrastructure using hfjobs we would simply need to instead run:

<!-- TODO update URLs once examples are published -->

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

## Understanding UV Scripts

### Script Header Format

### Dependency Declaration

### Python Version Requirements

## Running Scripts with UV and hfjobs

### Making the script available (TODO better name)

- Running a public script
- uploading script to HF

### Basic Command Pattern

### Choosing Docker Images

### Environment Setup

## Examples

### Example 1: Simple Script (CPU)

### Example 2: Data Processing with Dependencies

### Example 3: GPU Workload with ML Libraries

### Example 4: Production vLLM Example

## Best Practices

### Script Design for Cloud

### Error Handling

### Resource Management

## Common Patterns

### Data Input/Output

### Authentication

### Monitoring Progress

## Debugging and Troubleshooting

### Common Issues

### Testing Locally vs Cloud

## Reference

### Quick Command Templates

### Links to More Examples
