# Dataset Deduplication with hfjobs

Remove duplicate samples from datasets at scale using Hugging Face infrastructure.

## Overview

This example demonstrates how to deduplicate datasets using semantic similarity. Unlike exact matching, semantic deduplication identifies samples that have the same meaning even if worded differently.

## Use Cases

- **Clean training data**: Remove redundant samples that can lead to overfitting
- **Prevent train/test leakage**: Ensure no semantic overlap between splits
- **Improve data quality**: Remove near-duplicates while preserving diversity

## Available Scripts

### semantic-dedupe.py

Uses [SemHash](https://github.com/MinishLab/semhash) for semantic deduplication. Supports multiple methods:

- `deduplicate`: Remove semantic duplicates (default)
- `filter_outliers`: Remove anomalous samples
- `find_representative`: Select diverse representative samples

## Running on HF Infrastructure

### Prerequisites

<!-- TODO make sure we are always using the same approach to tokens so we don't confuse users -->

```bash
export HF_TOKEN=$(python -c "from huggingface_hub import HfFolder; print(HfFolder.get_token())")
```

### Basic Usage

<!-- TODO update URL to GitHub repo (for now) -->

```bash
hfjobs run --secret HF_TOKEN=$HF_TOKEN \
  ghcr.io/astral-sh/uv:debian-slim /bin/bash -c \
  "uv run https://huggingface.co/datasets/{username}/my-scripts/raw/main/semantic-dedupe.py \
  <dataset_id> <column> <output_repo>"
```

### Examples

**Small dataset (<100k samples)**:

```bash
hfjobs run --secret HF_TOKEN=$HF_TOKEN \
  ghcr.io/astral-sh/uv:debian-slim /bin/bash -c \
  "uv run https://huggingface.co/datasets/davanstrien/hfjobs-examples/raw/main/semantic-dedupe.py \
  imdb text davanstrien/imdb-deduplicated"
```

**Large dataset (use cpu-upgrade)**:

```bash
hfjobs run --flavor cpu-upgrade --secret HF_TOKEN=$HF_TOKEN \
  ghcr.io/astral-sh/uv:debian-slim /bin/bash -c \
  "uv run https://huggingface.co/datasets/davanstrien/hfjobs-examples/raw/main/semantic-dedupe.py \
  nvidia/Nemotron-Personas persona davanstrien/Personas-deduplicated"
```

**With custom threshold**:

```bash
hfjobs run --secret HF_TOKEN=$HF_TOKEN \
  ghcr.io/astral-sh/uv:debian-slim /bin/bash -c \
  "uv run https://huggingface.co/datasets/davanstrien/hfjobs-examples/raw/main/semantic-dedupe.py \
  squad question davanstrien/squad-dedup --threshold 0.9"
```

**Filter outliers instead**:

```bash
hfjobs run --secret HF_TOKEN=$HF_TOKEN \
  ghcr.io/astral-sh/uv:debian-slim /bin/bash -c \
  "uv run https://huggingface.co/datasets/davanstrien/hfjobs-examples/raw/main/semantic-dedupe.py \
  ag_news text davanstrien/ag-news-filtered --method filter_outliers"
```

## Performance Tips

1. **Test with small samples first**: Use `--max-samples 1000` to verify your setup
2. **Choose appropriate thresholds**: Lower = more aggressive deduplication
3. **Monitor progress**: Use `hfjobs logs <job_id>` to track progress

## Output

The script creates a new dataset repository with:

- Deduplicated dataset in parquet format
- Dataset card with deduplication statistics
- Metadata about the deduplication process

Example output repository: [davanstrien/imdb-deduplicated](https://huggingface.co/datasets/davanstrien/imdb-deduplicated)

## Cost Optimization

- Semantic deduplication is CPU-bound (embedding generation)
- GPU not required unless using custom embedding models
- For very large datasets (>10M), consider chunking the process
