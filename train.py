import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import get_cosine_schedule_with_warmup

from dataset import TASK_TYPES, BEVDataset, collate_fn, load_questions
from inference import autocast, evaluate, get_device, to_device
from logger import RunLogger
from network import build_network


def make_loader(split, config, device, shuffle):
    data = config["data"]
    max_questions = data["max_train_questions"] if split == "train" else data["max_test_questions"]
    dataset = BEVDataset(load_questions(split, max_questions, config["seed"]), data["max_state_tokens"], data["max_choice_tokens"])
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=data["batch_size"],
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=data["num_workers"],
        persistent_workers=data["num_workers"] > 0,  # keep workers alive across epochs and validation passes
        pin_memory=device.type == "cuda",
    )


def validate(network, test_loader, device, logger, step, best_accuracy):
    metrics = evaluate(network, test_loader, device)
    logger.log(step, split="test", **metrics)
    if metrics["accuracy"] > best_accuracy:
        best_accuracy = metrics["accuracy"]
        logger.save_checkpoint(network, name="best", step=step)
    return metrics, best_accuracy


def train(config, run_name):
    torch.manual_seed(config["seed"])
    device = get_device()
    logger = RunLogger(config, run_name, device)
    model, lora, optim, log = config["model"], config["lora"], config["optim"], config["logging"]

    head_config = {**config["head"], "num_task_types": len(TASK_TYPES)}
    network = build_network(model["name"], model["num_layers"], lora["r"], lora["alpha"], lora["dropout"],
                            lora["last_k_layers"], head_config).to(device)
    count = lambda params: sum(p.numel() for p in params if p.requires_grad)
    print(f"trainable params: backbone={count(network.backbone.parameters()):,} head={count(network.head.parameters()):,}")

    train_loader = make_loader("train", config, device, shuffle=True)
    test_loader = None
    if config["data"]["max_test_questions"] != 0:
        test_loader = make_loader("test", config, device, shuffle=False)

    lora_params = [p for p in network.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": float(optim["lr_lora"])},
        {"params": network.head.parameters(), "lr": float(optim["lr_head"])},
    ], weight_decay=float(optim["weight_decay"]))
    total_steps = optim["epochs"] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(optim["warmup_ratio"] * total_steps), total_steps)

    step = 0
    best_accuracy = -1.0
    network.train()
    for epoch in range(optim["epochs"]):
        epoch_loss, epoch_correct, epoch_count = 0.0, 0, 0
        for batch in train_loader:
            batch = to_device(batch, device)
            with autocast(device):
                logits = network(**batch)
            loss = F.cross_entropy(logits, batch["labels"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), optim["grad_clip"])
            optimizer.step()
            scheduler.step()
            step += 1

            correct = (logits.argmax(-1) == batch["labels"]).sum().item()
            epoch_loss += loss.item() * len(logits)
            epoch_correct += correct
            epoch_count += len(logits)

            if step % log["log_every"] == 0:
                lrs = {"lr_head": scheduler.get_last_lr()[1]}
                if lora_params:
                    lrs["lr_lora"] = scheduler.get_last_lr()[0]
                logger.log(step, split="train", epoch=epoch, loss=loss.item(), accuracy=correct / len(logits), **lrs)
            if step % log["save_every"] == 0:
                logger.save_checkpoint(network, name="latest", step=step)
            if test_loader is not None and step % log["eval_every"] == 0:
                test_metrics, best_accuracy = validate(network, test_loader, device, logger, step, best_accuracy)

        logger.log(step, split="train_epoch", epoch=epoch, loss=epoch_loss / epoch_count,
                   accuracy=epoch_correct / epoch_count)

    summary = {"final_train_loss": epoch_loss / epoch_count, "final_train_accuracy": epoch_correct / epoch_count}
    if test_loader is not None:
        if step % log["eval_every"] != 0:
            test_metrics, best_accuracy = validate(network, test_loader, device, logger, step, best_accuracy)
        summary["test"] = test_metrics
        summary["best_test_accuracy"] = best_accuracy

    ckpt_dir = logger.save_checkpoint(network, step=step)
    logger.finish(checkpoint=str(ckpt_dir), **summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML config, e.g. configs/smoke.yaml")
    parser.add_argument("--name", help="run name, used as runs/<name>-<timestamp> (default: config file name)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config, args.name or Path(args.config).stem)


if __name__ == "__main__":
    main()
