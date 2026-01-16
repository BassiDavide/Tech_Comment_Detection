import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import numpy as np

IGNORE_LABELS_RAW = {
    "manipulative wording: sarcasm",
    "other (unspecified)",
}

LABEL_ALIASES: Dict[str, str] = {
    "attack on reputation: name calling or labeling": "name calling",
    "attack on reputation: casting doubt": "smears/doubt",
    "attack on reputation: questioning the reputation (smears/poisoning the well)": "smears/doubt",
    "attack on reputation: guilt by association (reductio ad hitlerum)": "reductio ad hitlerum",
    "attack on reputation: glittering generalities": "name calling",
    "attack on reputation: appeal to hypocrisy (to quoque)": "appeal to hypocrisy (to quoque)",
    "justification: appeal to values/flag waving": "appeal to values/flag waving",
    "justification: appeal to values": "appeal to values/flag waving",
    "justification: flag waving": "appeal to values/flag waving",
    "justification: appeal to fear, prejudice": "appeal to fear, prejudice",
    "justification: appeal to pity": "appeal to pity",
    "justification: appeal to authority": "appeal to authority",
    "justification: appeal to popularity (bandwagon)": "appeal to popularity (bandwagon)",
    "manipulative wording: exaggeration or minimisation": "exaggeration or minimisation",
    "manipulative wording: loaded language": "loaded language",
    "manipulative wording: obfuscation, intentional vagueness, confusion": "intentional vagueness",
    "manipulative wording: repetition": "repetition",
    "manipulative wording": "loaded language",
    "simplification: causal oversimplification": "causal oversimplification",
    "simplification: consequential oversimplification (slippery slope)": "causal oversimplification",
    "simplification: false dilemma or no choice (black-and-white fallacy, dictatorship)": "black-and-white fallacy",
    "call: slogans": "slogans",
    "call: appeal to time (kairos)": "appeal to time",
    "call: conversation killer (thought-terminating cliché)": "thought-terminating cliché",
    "call": "slogans",
    "distraction: switching topic (whataboutism)": "distraction",
    "distraction: introducing irrelevant information (red herring)": "distraction",
    "distraction: misrepresentation of someone’s position (strawman)": "distraction",
}

IGNORE_LABELS = {label.casefold() for label in IGNORE_LABELS_RAW}
LABEL_ALIASES_CASEFOLD = {key.casefold(): value for key, value in LABEL_ALIASES.items()}


class MultiHeadDataProcessor:
    """Handles data loading and preprocessing for multi-head token labels."""

    def __init__(self, label_list: List[str], none_label: str = "none"):
        self.label_list = label_list
        self.num_labels = len(label_list)
        self.label2id = {label: idx for idx, label in enumerate(label_list)}
        self.canonical_by_cf = {label.casefold(): label for label in label_list}
        self.none_label = none_label
        self.none_index = self.label2id.get(none_label)
        self.technique_mask = np.ones(self.num_labels, dtype=bool)
        if self.none_index is not None:
            self.technique_mask[self.none_index] = False

    def load_data(self, file_path: str) -> List[Dict]:
        """Load JSONL data file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        return data

    def normalize_technique(self, raw: str) -> Optional[str]:
        """
        Apply alias/ignore rules and map to the canonical label_list entry.
        Returns None if the label should be skipped.
        """
        if not raw:
            return None
        cleaned = raw.strip()
        key = cleaned.casefold()
        if key in IGNORE_LABELS:
            return None
        normalized = LABEL_ALIASES_CASEFOLD.get(key, cleaned)
        return self.canonical_by_cf.get(normalized.casefold())

    def prepare_example(self, item: Dict) -> Tuple[str, List[Dict]]:
        """
        Extract text and normalized annotations from a single example.
        """
        text = item["CommentText"]
        annotations = item.get("annotations", [])

        normalized_annotations = []
        for ann in annotations:
            norm = self.normalize_technique(ann.get("technique"))
            if norm is None:
                continue
            if norm == self.none_label:
                continue
            normalized_annotations.append({**ann, "technique": norm})

        return text, normalized_annotations

    def create_token_labels(
        self,
        text: str,
        annotations: List[Dict],
        tokenizer,
        max_length: int,
    ) -> np.ndarray:
        """
        Create multi-label token annotations with an explicit "none" head.
        """
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offset_mapping = encoding["offset_mapping"][0].numpy()
        token_labels = np.zeros((max_length, self.num_labels), dtype=np.float32)

        for ann in annotations:
            technique = ann.get("technique")
            label_idx = self.label2id.get(technique)
            if label_idx is None:
                continue
            start_char, end_char = ann["start"], ann["end"]

            for idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start == 0 and token_end == 0:
                    continue
                if token_start < end_char and token_end > start_char:
                    token_labels[idx, label_idx] = 1.0

        if self.none_index is not None:
            valid_tokens = ~(
                (offset_mapping[:, 0] == 0) & (offset_mapping[:, 1] == 0)
            )
            has_label = token_labels[:, self.technique_mask].sum(axis=1) > 0
            token_labels[valid_tokens & ~has_label, self.none_index] = 1.0

        return token_labels


class MultiHeadDataset(Dataset):
    """PyTorch Dataset for multi-head token classification."""

    def __init__(
        self,
        texts: List[str],
        token_labels: List[np.ndarray],
        tokenizer,
        max_length: int = 256,
    ):
        self.texts = texts
        self.token_labels = token_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        token_label = self.token_labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_labels": torch.FloatTensor(token_label),
        }


def prepare_datasets(
    train_file: str,
    dev_file: str,
    test_file: str,
    label_list: List[str],
    tokenizer,
    max_length: int = 256,
):
    """
    Prepare train/dev/test datasets for multi-head span detection.
    """
    processor = MultiHeadDataProcessor(label_list)

    def build_dataset(split_data: List[Dict]):
        texts = []
        token_labels = []
        for item in split_data:
            text, annotations = processor.prepare_example(item)
            texts.append(text)
            labels = processor.create_token_labels(text, annotations, tokenizer, max_length)
            token_labels.append(labels)
        return texts, token_labels

    train_data = processor.load_data(train_file)
    dev_data = processor.load_data(dev_file)
    test_data = processor.load_data(test_file)

    train_texts, train_labels = build_dataset(train_data)
    dev_texts, dev_labels = build_dataset(dev_data)
    test_texts, test_labels = build_dataset(test_data)

    datasets = {
        "train": MultiHeadDataset(train_texts, train_labels, tokenizer, max_length),
        "dev": MultiHeadDataset(dev_texts, dev_labels, tokenizer, max_length),
        "test": MultiHeadDataset(test_texts, test_labels, tokenizer, max_length),
    }

    print(f"TRAIN - Examples: {len(train_texts)}")
    print(f"DEV - Examples: {len(dev_texts)}")
    print(f"TEST - Examples: {len(test_texts)}")

    return datasets["train"], datasets["dev"], datasets["test"], processor
