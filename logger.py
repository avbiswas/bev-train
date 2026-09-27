import json
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from network import BEVNetwork, ChoiceHead, load_backbone


class RunLogger:
    """Each training run gets runs/<name>-<timestamp>/ with config, metrics and checkpoints (LoRA + head only)."""

    def __init__(self, config, run_name, device, root="runs"):
        self.config = config
        self.exp_id = f"{run_name}-{datetime.now():%Y%m%d-%H%M%S}"
        self.run_dir = Path(root) / self.exp_id
        self.run_dir.mkdir(parents=True)
        self.start_time = time.time()

        self.write_config(self.run_dir)
        with open(self.run_dir / "run.json", "w") as f:
            json.dump({"exp_id": self.exp_id, "device": str(device), "started_at": datetime.now().isoformat()}, f, indent=2)
        print(f"Experiment {self.exp_id} -> {self.run_dir}")

    def write_config(self, directory):
        with open(directory / "config.yaml", "w") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)

    def log(self, step, **metrics):
        record = {"step": step, "time": round(time.time() - self.start_time, 1), **metrics}
        with open(self.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))

    def save_checkpoint(self, network, name="final", step=None):
        """A checkpoint folder is self-contained: copy it anywhere and load_checkpoint() rebuilds the network."""
        ckpt_dir = self.run_dir / "checkpoints" / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(network.backbone, PeftModel):
            network.backbone.save_pretrained(ckpt_dir / "lora")  # adapter weights only, not Qwen itself
        torch.save({k: v.cpu() for k, v in network.head.state_dict().items()}, ckpt_dir / "head.pt")

        self.write_config(ckpt_dir)
        with open(ckpt_dir / "meta.json", "w") as f:
            json.dump({
                "exp_id": self.exp_id,
                "step": step,
                "model_name": self.config["model"]["name"],
                "num_layers": network.backbone.config.num_hidden_layers,
                "head_config": {**self.config["head"], "num_task_types": network.head.task_embedding.num_embeddings},
                "max_state_tokens": self.config["data"]["max_state_tokens"],
                "max_choice_tokens": self.config["data"]["max_choice_tokens"],
            }, f, indent=2)
        print(f"Saved checkpoint -> {ckpt_dir}")
        return ckpt_dir

    def finish(self, **summary):
        summary["duration_sec"] = round(time.time() - self.start_time, 1)
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def load_checkpoint(ckpt_dir, device):
    ckpt_dir = Path(ckpt_dir)
    with open(ckpt_dir / "meta.json") as f:
        meta = json.load(f)

    backbone = load_backbone(meta["model_name"], meta.get("num_layers"))
    if (ckpt_dir / "lora").exists():
        backbone = PeftModel.from_pretrained(backbone, ckpt_dir / "lora")
    head = ChoiceHead(hidden_dim=backbone.config.hidden_size, **meta["head_config"])
    head.load_state_dict(torch.load(ckpt_dir / "head.pt", map_location="cpu"))
    return BEVNetwork(backbone, head).to(device).eval(), meta
