import json

import torch
from datasets import load_dataset

from tokenization import encode_example, tokenizer

DATASET_NAME = "avbiswas/bev-decision-150K"
TASK_TYPES = {"choice": 0, "noul": 1, "score": 2}
TASK_NAMES = {v: k for k, v in TASK_TYPES.items()}


def question_to_choices(question):
    """Turns any of the 3 question types into a list of option strings (+ label index if present)."""
    if question["type"] == "choice":
        keys = list(question["criteria"])
        choices = [f"{key}: {question['criteria'][key]}" for key in keys]
        label = keys.index(question["label"]) if "label" in question else None
    elif question["type"] == "score":
        # Choices share position ids, so the level index is written into the text to keep the ordering
        choices = [f"{i}: {level}" for i, level in enumerate(question["criteria"])]
        label = question.get("label")
    else:  # noul -> P(Yes)
        choices = ["No", "Yes"]
        label = int(question["label"]) if "label" in question else None
    return choices, label


def explode_questions(rows):
    """One dataset row can hold many questions about the same state -> one output row per question."""
    out = {"state": [], "task_type": [], "instructions": [], "choices": [], "label": []}
    for state, questions_json in zip(rows["state"], rows["questions_json"]):
        for question in json.loads(questions_json).values():
            choices, label = question_to_choices(question)
            out["state"].append(state)
            out["task_type"].append(TASK_TYPES[question["type"]])
            out["instructions"].append(question["instructions"])
            out["choices"].append(choices)
            out["label"].append(label)
    return out


def load_questions(split, max_questions=None, seed=0):
    ds = load_dataset(DATASET_NAME, split=split).shuffle(seed=seed)
    if max_questions is not None:
        # Every row has at least one question, so this many rows is always enough
        ds = ds.select(range(min(max_questions, len(ds))))
    ds = ds.map(explode_questions, batched=True, remove_columns=ds.column_names)
    if max_questions is not None:
        ds = ds.select(range(min(max_questions, len(ds))))
    return ds


class BEVDataset(torch.utils.data.Dataset):
    def __init__(self, questions, max_state_tokens, max_choice_tokens):
        self.questions = questions
        self.max_state_tokens = max_state_tokens
        self.max_choice_tokens = max_choice_tokens

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        row = self.questions[idx]
        example = encode_example(
            row["state"], row["instructions"], row["choices"], TASK_NAMES[row["task_type"]],
            self.max_state_tokens, self.max_choice_tokens,
        )
        example["task_type"] = row["task_type"]
        example["label"] = row["label"]
        return example


def collate_fn(batch):
    B = len(batch)
    L = max(len(ex["input_ids"]) for ex in batch)
    C = max(len(ex["choice_read_idx"]) for ex in batch)

    input_ids = torch.full((B, L), tokenizer.pad_token_id, dtype=torch.long)
    position_ids = torch.zeros((B, L), dtype=torch.long)
    # Padded query rows attend only to themselves, otherwise their softmax row is all -inf -> NaN
    attention_mask = torch.eye(L).repeat(B, 1, 1)
    choice_read_idx = torch.zeros((B, C), dtype=torch.long)
    choice_mask = torch.zeros((B, C), dtype=torch.bool)
    answer_end_idx = torch.zeros(B, dtype=torch.long)

    for b, ex in enumerate(batch):
        n, c = len(ex["input_ids"]), len(ex["choice_read_idx"])
        input_ids[b, :n] = torch.tensor(ex["input_ids"])
        position_ids[b, :n] = torch.tensor(ex["position_ids"])
        attention_mask[b, :n, :n] = ex["attention_mask"]
        choice_read_idx[b, :c] = torch.tensor(ex["choice_read_idx"])
        choice_mask[b, :c] = True
        answer_end_idx[b] = ex["answer_end_idx"]

    attention_mask = torch.where(attention_mask == 0, -float("inf"), 0.0).unsqueeze(1)  # B, 1, L, L

    out = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "choice_read_idx": choice_read_idx,
        "choice_mask": choice_mask,
        "answer_end_idx": answer_end_idx,
        "task_type": torch.tensor([ex["task_type"] for ex in batch], dtype=torch.long),
    }
    if all(ex["label"] is not None for ex in batch):
        out["labels"] = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
    return out
