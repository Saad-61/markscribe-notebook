# MarkScribe

MarkScribe is a vision-language notebook that fine-tunes Qwen2-VL-2B-Instruct with QLoRA to transcribe document images into clean Markdown.

The notebook is built around the Nougat document image to Markdown corpus and is designed to run on resource-constrained GPUs (for example, T4 16 GB) using 4-bit quantization plus LoRA adapters.

## What this notebook includes

- End-to-end pipeline for data loading, preprocessing, and training.
- QLoRA fine-tuning configuration for Qwen2-VL-2B-Instruct.
- Validation generation and qualitative comparison against ground truth.
- Visual analysis cells for predictions and sample pages.
- Bonus experiments (epoch comparison, prompt ablation, and zero-shot vs fine-tuned checks).
- Gradio demo cell for interactive image-to-Markdown inference.

## Repository contents

- markscribe.ipynb: Full training and inference workflow.

## Model and training setup

- Base model: Qwen/Qwen2-VL-2B-Instruct
- Fine-tuning method: QLoRA (NF4, 4-bit)
- Typical LoRA settings: rank 16, alpha 32
- Default split: 80/20 train and validation
- Default training profile: lightweight settings intended for limited GPU memory

## Dataset

The notebook targets the Nougat training dataset example and expects a JSONL index containing image paths and Markdown targets.

Expected layout (simplified):

- nougat-training-dataset-example/
- 0508.jsonl
- 0508/<paper-id>/<page>.png

The notebook automatically checks common Kaggle and local paths, then resolves image files from the JSONL entries.

## Environment setup

Recommended: Python 3.10+ with CUDA-enabled PyTorch.

Core dependencies used in the notebook:

- transformers
- accelerate
- peft
- bitsandbytes
- datasets
- qwen-vl-utils
- trl
- sentencepiece
- matplotlib
- tqdm
- gradio
- pillow

If you run in Kaggle, the notebook already includes install cells for compatible versions.

## How to run

1. Open markscribe.ipynb in VS Code, Jupyter Lab, or Kaggle.
2. Run environment setup cells first.
3. Verify dataset path detection in the data exploration section.
4. Run training cells to produce LoRA adapters.
5. Run generation and visualization cells to inspect model output quality.
6. Optionally run the Gradio section for interactive inference.

## Output artifacts

During execution, the notebook may create:

- Adapter checkpoints
- Training logs and loss snapshots
- Sample prediction outputs
- Visualization plots

## Notes

- The notebook is optimized for experimentation and educational reproducibility.
- Runtime and quality depend heavily on GPU memory, dataset subset size, and epoch count.
