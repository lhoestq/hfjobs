# UV Script Sharing with hfjobs

This guide explains how to share UV scripts on the Hugging Face Hub using the new `hfjobs scripts` commands.

## Overview

The `hfjobs scripts` commands provide a simple way to:
- Share UV scripts as Hugging Face dataset repositories
- Make scripts easily discoverable with standardized tags
- Generate usage instructions automatically
- Run shared scripts with a simple copy-paste command

## Quick Start

### 1. Share a UV Script

To share an existing UV script:

```bash
hfjobs scripts init my-awesome-script my-script.py
```

This will:
- Create a new dataset repository (e.g., `username/my-awesome-script`)
- Upload your UV script
- Generate a README with usage instructions
- Tag the repository with `hfjobs-uv-script` for discovery

### 2. Create a Template Script

If you don't have a script yet, omit the script argument to create a template:

```bash
hfjobs scripts init my-new-script
```

This creates a repository with a template UV script that you can customize.

### 3. Add More Scripts

To add additional scripts to an existing repository:

```bash
hfjobs scripts push another-script.py
```

The README will be automatically updated with the new script.

## Running Shared Scripts

Once a script is shared, anyone can run it using the command shown in the repository README:

```bash
hfjobs run ghcr.io/astral-sh/uv:python3.12 \
  uv run https://huggingface.co/datasets/username/my-script/resolve/main/script.py \
  <arguments>
```

## Example UV Script

Here's a simple UV script that filters a dataset by text length:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "pandas",
# ]
# ///
"""Filter dataset by text length."""

import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Filter dataset by text length")
    parser.add_argument("input_dataset", help="Input dataset from HF Hub")
    parser.add_argument("output_dataset", help="Output dataset name")
    parser.add_argument("--min-length", type=int, default=10, help="Minimum text length")
    
    args = parser.parse_args()
    
    dataset = load_dataset(args.input_dataset, split="train")
    filtered = dataset.filter(lambda x: len(x["text"]) >= args.min_length)
    filtered.push_to_hub(args.output_dataset)
    print("✅ Done!")

if __name__ == "__main__":
    main()
```

## Repository Structure

A UV script repository has a simple structure:

```
username/my-script/
├── script.py          # Your UV script(s)
└── README.md         # Auto-generated usage instructions
```

## Discovery

All scripts shared with `hfjobs scripts init` are automatically tagged with:
- `hfjobs-uv-script`
- `uv`
- `python`

This makes them easy to find on the Hugging Face Hub.

## Best Practices

1. **Use descriptive repository names** - Make it clear what your script does
2. **Add docstrings** - The first line becomes the description in the README
3. **Include usage examples** - Add examples in your script's docstring
4. **Specify dependencies clearly** - Use the UV script header format
5. **Test locally first** - Ensure your script works before sharing

## Private Scripts

To create a private repository for internal use:

```bash
hfjobs scripts init my-private-script script.py --private
```

## Tips

- The last initialized repository is remembered, so you can use `hfjobs scripts push` without specifying `--repo`
- Script dependencies are automatically extracted and shown in the README
- Multiple scripts in one repository are supported and organized in the README

## Next Steps

- Browse existing UV scripts: Search for the `hfjobs-uv-script` tag on [Hugging Face Hub](https://huggingface.co/datasets)
- Share your own scripts to help the community
- Contribute improvements to the [hfjobs repository](https://github.com/huggingface/hfjobs)