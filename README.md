# Training JEV networks from Qwen

This repo is under construction. This is for the upcoming Neural Breakdown video on training JEV networks from existing autoregressive LMs. The cool part of this approach is that we are making the architecture be choice-order invariant, while still passing multiple options in the same sequence.

## Start here: `arch.ipynb`

If you want to understand how this repo works, open [`arch.ipynb`](arch.ipynb) first. It's the notebook from the livestream, and it builds the core idea step by step: tokenizing the question and each choice separately, giving every choice the same starting position ids, and building the attention mask so no choice can see another. Nothing gets trained in there, it's all about the architecture.

The Python files take that notebook code and turn it into a full pipeline. `tokenization.py` reuses the notebook functions almost as they are, and `dataset.py`, `network.py`, `train.py` and `inference.py` build the rest around them.

Dataset: https://huggingface.co/datasets/avbiswas/bev-decision-150K

Livestream: https://youtube.com/live/AzxoU7kxjig

