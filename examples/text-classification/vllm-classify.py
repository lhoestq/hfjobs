# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "httpx",
#     "huggingface-hub",
#     "setuptools",
#     "toolz",
#     "transformers",
#     "vllm",
# ]
#
# [[tool.uv.index]]
# url = "https://wheels.vllm.ai/nightly"
# ///

import logging
import os
from typing import Optional

import httpx
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_url, login
from toolz import concat, partition_all, keymap
from tqdm.auto import tqdm
from vllm import LLM
import vllm
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# log vllm version
print(vllm.__version__)


def get_model_id2label(hub_model_id: str) -> Optional[dict[str, str]]:
    response = httpx.get(
        hf_hub_url(
            hub_model_id,
            filename="config.json",
        )
    )
    if response.status_code != 200:
        return None
    try:
        data = response.json()
        logger.info(f"Config: {data}")
        id2label = data.get("id2label")
        if id2label is None:
            logger.error("id2label is not found in config.json")
            return None
        return keymap(int, id2label)
    except Exception as e:
        logger.error(f"Failed to parse config.json: {e}")
        return None


def get_top_label(output, label_map: Optional[dict[str, str]] = None):
    """
    Given a ClassificationRequestOutput and a label_map (e.g. {'0': 'label0', ...}),
    returns the top predicted label (or None if not found) and its confidence score.
    """
    logits = torch.tensor(output.outputs.probs)
    probs = F.softmax(logits, dim=0)
    top_idx = torch.argmax(probs).item()
    top_prob = probs[top_idx].item()
    label = label_map.get(top_idx) if label_map is not None else top_idx
    return label, top_prob


def format_prompts(dataset, inference_column, inference_columns, prompt_template, column_separator):
    """Format prompts based on the provided arguments."""
    
    if inference_columns:
        # Multiple columns specified
        columns = [col.strip() for col in inference_columns.split(',')]
        
        # Validate columns exist
        for col in columns:
            if col not in dataset.column_names:
                raise ValueError(f"Column '{col}' not found in dataset. Available: {dataset.column_names}")
        
        if prompt_template:
            # Use template formatting
            prompts = []
            for row in dataset:
                format_dict = {col: row[col] for col in columns}
                try:
                    # Replace \\n with actual newlines in the template
                    template = prompt_template.replace('\\n', '\n')
                    prompt = template.format(**format_dict)
                    prompts.append(prompt)
                except KeyError as e:
                    raise ValueError(f"Template placeholder {e} not found in columns: {columns}")
        else:
            # Join columns with separator
            prompts = [
                column_separator.join(str(row[col]) for col in columns)
                for row in dataset
            ]
    else:
        # Single column (backward compatible)
        if inference_column not in dataset.column_names:
            raise ValueError(f"Column '{inference_column}' not found in dataset")
        prompts = dataset[inference_column]
    
    return prompts


def main(
    hub_model_id: str,
    src_dataset_hub_id: str,
    output_dataset_hub_id: str,
    inference_column: str = "text",
    inference_columns: Optional[str] = None,
    prompt_template: Optional[str] = None,
    column_separator: str = " ",
    batch_size: int = 10_000,
    hf_token: Optional[str] = None,
):
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN is not None:
        login(token=HF_TOKEN)
    else:
        raise ValueError("HF_TOKEN is not set")
    llm = LLM(model=hub_model_id, task="classify")
    id2label = get_model_id2label(hub_model_id)
    dataset = load_dataset(src_dataset_hub_id, split="train")
    
    # Format prompts based on arguments
    prompts = format_prompts(dataset, inference_column, inference_columns, prompt_template, column_separator)
    logger.info(f"Formatted {len(prompts)} prompts")
    if prompts:
        logger.info(f"Example prompt: {prompts[0][:200]}...")
    all_results = []
    all_results.extend(
        llm.classify(batch) for batch in tqdm(list(partition_all(batch_size, prompts)))
    )
    outputs = list(concat(all_results))
    if id2label is not None:
        labels_and_probs = [get_top_label(output, id2label) for output in outputs]
        dataset = dataset.add_column("label", [label for label, _ in labels_and_probs])
        dataset = dataset.add_column("prob", [prob for _, prob in labels_and_probs])
    else:
        # just append raw label index and probs
        dataset = dataset.add_column(
            "label", [output.outputs.label for output in outputs]
        )
        dataset = dataset.add_column(
            "prob", [output.outputs.probs for output in outputs]
        )
    dataset.push_to_hub(output_dataset_hub_id, token=HF_TOKEN)
    
    # Create and push dataset card
    from huggingface_hub import DatasetCard
    
    card_content = f"""---
tags:
- text-classification
- vllm
---

# {output_dataset_hub_id}

This dataset was created by classifying [{src_dataset_hub_id}](https://huggingface.co/datasets/{src_dataset_hub_id}) 
using [{hub_model_id}](https://huggingface.co/{hub_model_id}).

## Prompt Format
"""
    
    if inference_columns:
        card_content += f"Columns used: `{inference_columns}`\n\n"
        if prompt_template:
            card_content += f"Template:\n```\n{prompt_template}\n```\n\n"
        else:
            card_content += f"Columns joined with: `{column_separator}`\n\n"
    else:
        card_content += f"Column used: `{inference_column}`\n\n"
    
    if id2label:
        card_content += f"\n## Labels\n\n{', '.join([f'`{label}`' for label in id2label.values()])}\n"
    
    card_content += f"\n## Processing Details\n\n- Batch size: {batch_size:,}\n- Date: {os.popen('date').read().strip()}\n"
    
    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset_hub_id, repo_type="dataset", token=HF_TOKEN)
    logger.info(f"Dataset and card pushed to: https://huggingface.co/datasets/{output_dataset_hub_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Classify a dataset using a Hugging Face model and save results to Hugging Face Hub",
    )
    parser.add_argument(
        "hub_model_id",
        type=str,
        help="Hugging Face model ID to use for classification",
    )
    parser.add_argument(
        "src_dataset_hub_id",
        type=str,
        help="Source dataset ID on Hugging Face Hub",
    )
    parser.add_argument(
        "output_dataset_hub_id",
        type=str,
        help="Output dataset ID on Hugging Face Hub",
    )
    parser.add_argument(
        "--inference-column",
        type=str,
        default="text",
        help="Column name containing text to classify (default: text)",
    )
    parser.add_argument(
        "--inference-columns",
        type=str,
        help="Comma-separated list of columns to combine (e.g., 'title,abstract')"
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        help="Template string with placeholders (e.g., 'Title: {title}\\nAbstract: {abstract}')"
    )
    parser.add_argument(
        "--column-separator",
        type=str,
        default=" ",
        help="Separator when joining columns without template (default: space)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Batch size for inference (default: 10000)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token (default: None)",
    )

    args = parser.parse_args()
    main(
        hub_model_id=args.hub_model_id,
        src_dataset_hub_id=args.src_dataset_hub_id,
        output_dataset_hub_id=args.output_dataset_hub_id,
        inference_column=args.inference_column,
        inference_columns=args.inference_columns,
        prompt_template=args.prompt_template,
        column_separator=args.column_separator,
        batch_size=args.batch_size,
        hf_token=args.hf_token,
    )

# hfjobs run --flavor l4x1 \
#         --secret HF_TOKEN=hf_*** \
#         ghcr.io/astral-sh/uv:debian \
#         /bin/bash -c "
#       export HOME=/tmp && \
#       export USER=dummy && \
#       export TORCHINDUCTOR_CACHE_DIR=/tmp/torch-inductor && \
#       uv run https://huggingface.co/datasets/davanstrien/dataset-creation-scripts/raw/main/vllm-bert-classify-dataset/main.py \
#         davanstrien/ModernBERT-base-is-new-arxiv-dataset \
#         davanstrien/testarxiv \
#         davanstrien/testarxiv-out \
#         --inference-column prompt \
#         --batch-size 100000" \
#         --project vllm-classify \
#         --name testarxiv-classify
