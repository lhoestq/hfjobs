# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "semhash",
#     "datasets",
#     "huggingface-hub",
#     "hf-transfer",
#     "hf-xet",
# ]
# ///
"""Deduplicate a Hugging Face dataset using SemHash.

This script uses semantic deduplication to remove duplicate entries from a dataset
based on a specified text column, then pushes the results to a new dataset repository.
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Optional

from datasets import Dataset, load_dataset
from huggingface_hub import DatasetCard
from semhash import SemHash
from huggingface_hub import login

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = (
    "1"  # Enable HF transfer to speed up transfers
)
HF_TOKEN = os.environ.get("HF_TOKEN", None)  # Get Hugging Face token from environment
assert HF_TOKEN, "HF_TOKEN environment variable must be set for authentication"
login(HF_TOKEN)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deduplicate a Hugging Face dataset using semantic similarity"
    )
    parser.add_argument(
        "dataset_id",
        type=str,
        help="Source dataset ID (e.g., 'imdb', 'squad', 'username/dataset-name')",
    )
    parser.add_argument(
        "column",
        type=str,
        help="Column name to deduplicate on (e.g., 'text', 'question', 'context')",
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="Target repository ID for deduplicated dataset (e.g., 'username/my-deduplicated-dataset')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process (default: train)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Similarity threshold for deduplication (0-1, default: auto)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["deduplicate", "filter_outliers", "find_representative"],
        default="deduplicate",
        help="Deduplication method to use (default: deduplicate)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the output dataset private",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )

    return parser.parse_args()


def create_dataset_card(
    original_dataset_id: str,
    column: str,
    method: str,
    duplicate_ratio: float,
    original_size: int,
    deduplicated_size: int,
    threshold: Optional[float] = None,
) -> str:
    """Create a dataset card with deduplication information."""
    card_content = f"""---
tags:
- deduplicated
- semhash
- semantic-deduplication
- hfjobs
---

# Deduplicated {original_dataset_id}

This dataset is a deduplicated version of [{original_dataset_id}](https://huggingface.co/datasets/{original_dataset_id}) 
using semantic deduplication with [SemHash](https://github.com/MinishLab/semhash).

## Deduplication Details

- **Method**: {method}
- **Column**: `{column}`
- **Original size**: {original_size:,} samples
- **Deduplicated size**: {deduplicated_size:,} samples
- **Duplicate ratio**: {duplicate_ratio:.2%}
- **Reduction**: {(1 - deduplicated_size / original_size):.2%}
"""

    if threshold is not None:
        card_content += f"- **Similarity threshold**: {threshold}\n"

    card_content += f"""
- **Date processed**: {datetime.now().strftime("%Y-%m-%d")}

## How to use

```python
from datasets import load_dataset

dataset = load_dataset("{original_dataset_id.split("/")[-1]}-deduplicated")
```

## Processing script

This dataset was created using the following script:

```bash
uv run dedupe-dataset.py {original_dataset_id} {column} <repo_id> --method {method}
```

## About semantic deduplication

Unlike exact deduplication, semantic deduplication identifies and removes samples that are 
semantically similar even if they use different words. This helps create cleaner training 
datasets and prevents data leakage between train/test splits.
"""

    return card_content


def main():
    """Main function to run deduplication."""
    args = parse_args()

    # Check for HF token
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "Warning: HF_TOKEN not found in environment. You may not be able to push to private repos."
        )

    # Load dataset
    print(f"Loading dataset '{args.dataset_id}' (split: {args.split})...")
    try:
        if args.max_samples:
            dataset = load_dataset(
                args.dataset_id, split=f"{args.split}[:{args.max_samples}]", token=token
            )
        else:
            dataset = load_dataset(args.dataset_id, split=args.split, token=token)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Validate column exists
    if args.column not in dataset.column_names:
        print(f"Error: Column '{args.column}' not found in dataset.")
        print(f"Available columns: {', '.join(dataset.column_names)}")
        sys.exit(1)

    # Convert dataset to records for semhash
    print(f"Preparing dataset for deduplication on column '{args.column}'...")
    records = [dict(row) for row in dataset]
    original_size = len(records)
    print(f"Found {original_size:,} samples")

    # Initialize SemHash with the specific column
    print("Initializing SemHash with default model...")
    semhash = SemHash.from_records(records=records, columns=[args.column])

    # Apply selected method
    print(f"Applying {args.method} method...")
    if args.method == "deduplicate":
        if args.threshold:
            result = semhash.self_deduplicate(threshold=args.threshold)
        else:
            result = semhash.self_deduplicate()
    elif args.method == "filter_outliers":
        result = semhash.self_filter_outliers()
    elif args.method == "find_representative":
        result = semhash.self_find_representative()

    # Get deduplicated records
    deduplicated_records = result.selected
    deduplicated_size = len(deduplicated_records)

    # Print statistics
    print("\nDeduplication complete!")
    print(f"Original size: {original_size:,}")
    print(f"Deduplicated size: {deduplicated_size:,}")
    print(
        f"Removed: {original_size - deduplicated_size:,} ({result.duplicate_ratio:.2%})"
    )

    # Create new dataset from deduplicated records
    print("\nCreating deduplicated dataset...")
    deduplicated_dataset = Dataset.from_list(deduplicated_records)

    # Push dataset to hub first (this creates the repo)
    print(f"\nPushing deduplicated dataset to '{args.repo_id}'...")
    try:
        deduplicated_dataset.push_to_hub(
            args.repo_id,
            private=args.private,
            token=token,
            commit_message=f"Add deduplicated version of {args.dataset_id}",
        )
        print("Dataset pushed successfully!")

        # Create and push dataset card
        print("Creating and pushing dataset card...")
        card_content = create_dataset_card(
            original_dataset_id=args.dataset_id,
            column=args.column,
            method=args.method,
            duplicate_ratio=result.duplicate_ratio,
            original_size=original_size,
            deduplicated_size=deduplicated_size,
            threshold=args.threshold,
        )

        card = DatasetCard(card_content)
        card.push_to_hub(
            repo_id=args.repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Add dataset card",
        )

        print(
            f"\nSuccess! Dataset available at: https://huggingface.co/datasets/{args.repo_id}"
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
