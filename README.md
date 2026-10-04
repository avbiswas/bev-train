[Watch the video: Training JEV networks from Qwen](https://youtu.be/sF3CNPbWA8o)

If you find this helpful, consider supporting on Patreon — it hosts all code, projects, slides, and write-ups from the YouTube channel.

[<img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patron!" width="200">](https://www.patreon.com/NeuralBreakdownwithAVB)

# Training JEV networks from Qwen

Train a network that answers choice, yes/no (`noul`), and score questions using a Qwen backbone and a small attention head. Choices share the same starting position IDs, and the attention mask prevents each choice from seeing the others. The head scores the choices without depending on their order.

## Start here: `arch.ipynb`

Open [`arch.ipynb`](arch.ipynb) for the architecture walkthrough from the livestream: tokenize the question and choices separately, build their position IDs, and construct the attention mask. The notebook explains the architecture; the Python files implement training and inference.

- [`tokenization.py`](tokenization.py): prompt formatting, choice positions, and attention masks.
- [`dataset.py`](dataset.py): expand dataset rows into typed questions and batch them.
- [`network.py`](network.py): Qwen backbone, optional LoRA, and the choice-scoring head.
- [`train.py`](train.py): training, gradient accumulation, validation, and checkpoint initialization.
- [`logger.py`](logger.py): run metrics and LoRA/head checkpoints.
- [`inference.py`](inference.py): evaluation and typed answers through `answer()`.

Dataset: [avbiswas/bev-decision-150K](https://huggingface.co/datasets/avbiswas/bev-decision-150K). Training loads this dataset automatically.

Original livestream: [Architecture walkthrough](https://youtube.com/live/AzxoU7kxjig).

## Setup

Install [uv](https://docs.astral.sh/uv/), then install the locked dependencies:

```bash
uv sync
```

The first run downloads the Qwen3-0.6B weights, tokenizer, and dataset. The code selects CUDA, then Apple MPS, then CPU. Full training configs are intended for a GPU; reduce the batch size for smaller devices.

For a fresh Ubuntu GPU machine, [`cloud_startup_script.sh`](cloud_startup_script.sh) installs the system tools, sets up uv, and checks CUDA availability:

```bash
bash cloud_startup_script.sh
```

## Train

For a small overfitting sanity check, use the 10-question config:

```bash
uv run python train.py configs/smoke.yaml --name smoke
```

This config runs 30 epochs to check whether the model can learn those questions. For full training with the first 20 Qwen layers and LoRA on the last 12 kept layers (8–19):

```bash
uv run python train.py configs/full_lora_k12.yaml --name full-lora-k12
```

That config trains the choice head and LoRA adapters, uses 1,024 state tokens, and accumulates two microbatches of 32 questions per optimizer step. `optim.grad_accum_steps` defaults to 1. Logging, validation, and saving intervals count optimizer steps. An incomplete final accumulation group is skipped.

Other configs include [`full_head_only.yaml`](configs/full_head_only.yaml) for a frozen backbone and [`full_lora.yaml`](configs/full_lora.yaml) for LoRA on the last four of 20 kept layers.

Each run writes its config, metrics, summary, and checkpoints under `runs/<name>-<timestamp>/`. Checkpoints contain the head and, when enabled, LoRA weights; the Qwen base weights are loaded separately. `best` tracks test accuracy, `latest` is overwritten at each save interval, and `final` is saved at the end.

## Start from saved weights

Pass a checkpoint directory or a run ID (which selects that run's final checkpoint):

```bash
uv run python train.py configs/full_lora_k12.yaml --name continued \
  --resume runs/<run-id>/checkpoints/final
```

Use the same backbone, head architecture, and LoRA configuration as the checkpoint. `--resume` initializes a new training run from the saved weights: the optimizer, learning-rate schedule, and step counter start fresh.

## Evaluate

```bash
uv run python inference.py runs/<run-id>/checkpoints/final \
  --max_questions 100 --batch_size 2 --num_workers 0
```

Evaluation reports loss, overall accuracy, accuracy per task type, and yes/no AUC when both label classes are present. It uses the checkpoint's state and choice token limits.
