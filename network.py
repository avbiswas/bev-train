import math

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Model


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, new_dim):
        super().__init__()
        self.new_dim = new_dim
        self.q = nn.Linear(hidden_dim, new_dim)
        self.k = nn.Linear(hidden_dim, new_dim)
        self.v = nn.Linear(hidden_dim, new_dim)
        self.o = nn.Linear(new_dim, hidden_dim)

    def forward(self, x, padding_mask):

        # x: [B, T, H], padding_mask: [B, T] (True = real token)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # compute attention weights, padded choices are never attended to
        scores = torch.einsum("bqd,bkd->bqk", [q, k])/math.sqrt(self.new_dim) # [B, T, T]
        scores = scores.masked_fill(~padding_mask[:, None, :], -float("inf"))
        weights = torch.softmax(scores, dim=-1)

        out = torch.einsum("bqk,bkd->bqd", [weights, v])

        return self.o(out)


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, new_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, new_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, new_dim),
            nn.GELU(),
            nn.Linear(new_dim, hidden_dim),
        )

    def forward(self, x, padding_mask):
        x = x + self.attn(self.norm1(x), padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ChoiceHead(nn.Module):
    """Runs self-attention over [task token, choice embeddings..., answer embedding] and scores each choice."""

    def __init__(self, hidden_dim, new_dim, num_layers, num_task_types):
        super().__init__()
        self.new_dim = new_dim
        self.task_embedding = nn.Embedding(num_task_types, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([AttentionBlock(hidden_dim, new_dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.answer_proj = nn.Linear(hidden_dim, new_dim)
        self.choice_proj = nn.Linear(hidden_dim, new_dim)

    def forward(self, choice_embeddings, answer_embedding, choice_mask, task_type):
        # choice_embeddings: [B, C, H], answer_embedding: [B, H], choice_mask: [B, C], task_type: [B]
        B = choice_embeddings.shape[0]
        task_token = self.task_embedding(task_type).unsqueeze(1)

        x = torch.cat([task_token, choice_embeddings, answer_embedding.unsqueeze(1)], dim=1) # [B, C+2, H]
        x = self.input_norm(x)

        always_valid = torch.ones(B, 1, dtype=torch.bool, device=choice_mask.device)
        padding_mask = torch.cat([always_valid, choice_mask, always_valid], dim=1)
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = self.final_norm(x)

        # Bilinear score between the answer slot and every choice slot -> works for any number of choices
        answer = self.answer_proj(x[:, -1])        # [B, D]
        choices = self.choice_proj(x[:, 1:-1])     # [B, C, D]
        logits = torch.einsum("bd,bcd->bc", [answer, choices])/math.sqrt(self.new_dim)
        return logits.masked_fill(~choice_mask, -float("inf"))


class BEVNetwork(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids, position_ids, attention_mask, choice_read_idx, choice_mask, answer_end_idx, task_type, **kwargs):
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False
        )
        hidden = output.last_hidden_state.float()
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)

        choice_embeddings = hidden[batch_idx[:, None], choice_read_idx]   # [B, C, H]
        answer_embedding = hidden[batch_idx, answer_end_idx]              # [B, H]
        # Head runs in fp32 even under bf16 autocast: bf16 logits are too coarse for a stable loss
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.head(choice_embeddings, answer_embedding, choice_mask, task_type)


def load_backbone(model_name, num_layers=None):
    # Only the first num_layers decoder layers are built and loaded; the rest are never executed
    overrides = {} if num_layers is None else {"num_hidden_layers": num_layers}
    return Qwen3Model.from_pretrained(model_name, dtype=torch.float32, **overrides)


def lora_target_regex(num_layers, last_k_layers):
    layer_ids = "|".join(str(i) for i in range(num_layers - last_k_layers, num_layers))
    return rf".*layers\.({layer_ids})\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"


def build_network(model_name, num_layers, lora_r, lora_alpha, lora_dropout, lora_last_k_layers, head_config):
    backbone = load_backbone(model_name, num_layers)
    if lora_last_k_layers == 0:  # frozen Qwen, only the head trains
        backbone.requires_grad_(False)
    else:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_regex(backbone.config.num_hidden_layers, lora_last_k_layers),
        )
        backbone = get_peft_model(backbone, lora_config)
    head = ChoiceHead(hidden_dim=backbone.config.hidden_size, **head_config)
    return BEVNetwork(backbone, head)
