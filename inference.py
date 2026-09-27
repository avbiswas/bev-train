import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F

from dataset import TASK_TYPES, BEVDataset, collate_fn, load_questions, question_to_choices
from logger import load_checkpoint
from tokenization import encode_example


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device):
    # bf16 keeps fp32's exponent range, so no loss scaling is needed (unlike fp16)
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type in ("cuda", "mps"))


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def binary_auc(scores, labels):
    """Probability that a random positive gets a higher score than a random negative (ties count half)."""
    scores, labels = torch.tensor(scores), torch.tensor(labels, dtype=torch.bool)
    positives, negatives = scores[labels], scores[~labels].sort().values
    below = torch.searchsorted(negatives, positives, right=False)
    below_or_tied = torch.searchsorted(negatives, positives, right=True)
    return (below + below_or_tied).sum().item() / 2 / (len(positives) * len(negatives))


@torch.no_grad()
def evaluate(network, dataloader, device):
    network.eval()
    total_loss, correct, count = 0.0, defaultdict(int), defaultdict(int)
    type_names = {v: k for k, v in TASK_TYPES.items()}
    noul_p_yes, noul_labels = [], []

    for batch in dataloader:
        batch = to_device(batch, device)
        with autocast(device):
            logits = network(**batch)
        total_loss += F.cross_entropy(logits, batch["labels"], reduction="sum").item()

        hits = logits.argmax(-1) == batch["labels"]
        for task_type, hit in zip(batch["task_type"].tolist(), hits.tolist()):
            count[type_names[task_type]] += 1
            correct[type_names[task_type]] += hit

        is_noul = batch["task_type"] == TASK_TYPES["noul"]
        noul_p_yes += torch.softmax(logits[is_noul].float(), dim=-1)[:, 1].tolist()
        noul_labels += batch["labels"][is_noul].tolist()

    n = sum(count.values())
    metrics = {"loss": total_loss / n, "accuracy": sum(correct.values()) / n}
    for name in count:
        metrics[f"accuracy_{name}"] = correct[name] / count[name]
    if 0 < sum(noul_labels) < len(noul_labels):
        metrics["auc_noul"] = binary_auc(noul_p_yes, noul_labels)
    network.train()
    return metrics


@torch.no_grad()
def answer(network, meta, state, question, device):
    """Answers one question dict (dataset / TypeSafe format) with a typed answer."""
    network.eval()
    choices, _ = question_to_choices(question)
    example = encode_example(state, question["instructions"], choices, question["type"],
                             meta["max_state_tokens"], meta["max_choice_tokens"])
    example.update(task_type=TASK_TYPES[question["type"]], label=None)
    batch = to_device(collate_fn([example]), device)

    with autocast(device):
        probs = torch.softmax(network(**batch)[0].float(), dim=-1).tolist()

    if question["type"] == "choice":
        keys = list(question["criteria"])
        return {"choice": keys[max(range(len(keys)), key=probs.__getitem__)], "probabilities": dict(zip(keys, probs))}
    if question["type"] == "score":
        return {"score": sum(i * p for i, p in enumerate(probs)), "probabilities": probs}
    return {"noul": probs[1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="runs/<exp_id>/checkpoints/<name>")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_questions", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    network, meta = load_checkpoint(args.checkpoint, device)
    questions = load_questions(args.split, args.max_questions)
    # Same truncation as during training, read from the checkpoint
    dataset = BEVDataset(questions, meta["max_state_tokens"], meta["max_choice_tokens"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                         num_workers=args.num_workers)
    print(evaluate(network, loader, device))


if __name__ == "__main__":
    main()
