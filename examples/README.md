# hfjobs Examples

Production-ready examples for running workloads on Hugging Face infrastructure.

## Available Examples

### [Dataset Deduplication](./dataset-deduplication/)

Remove duplicate samples from datasets using semantic similarity. Includes examples for cleaning training data and preventing train/test leakage.

### Coming Soon

- **Training** - Multi-node training examples
- **vLLM Inference** - Run optimized inference at scale
- **Synthetic Data Generation** - Generate high-quality synthetic datasets
- **Data Processing Pipelines** - ETL workflows for ML data

## Quick Start

1. **Install hfjobs**:

   ```bash
   pip install hfjobs
   ```

2. **Set your HF token**:

   ```bash
   export HF_TOKEN=$(python -c "from huggingface_hub import HfFolder; print(HfFolder.get_token())")
   ```

3. **Browse the examples** above for your use case

## Simple Examples

Looking for basic hfjobs usage? Check out [docs/examples/](../docs/examples/) for pedagogical examples focused on learning the basics.

## Contributing

To add a new example:

1. Create a task-focused directory (e.g., `model-quantization/`)
2. Include a comprehensive README with use cases and benchmarks
3. Provide runnable scripts with clear documentation
4. Add performance metrics and cost estimates

Each example should solve a real problem users face when scaling ML workloads.
