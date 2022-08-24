from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import torch_default_data_collator


def convert_entities_from_xml(
    source: str,
    entity_start_tokens: dict[Optional[str], str],
    entity_end_tokens: dict[Optional[str], str],
) -> tuple[str, list[str], Optional[list[str]]]:
    # Parse XML by wrapping the source with `<root>` tag and extract the entity names.
    root = ET.fromstringlist(["<root>", source, "</root>"])
    names = [element.attrib["name"] for element in root]

    # While all entities are wrapped with `<entity>` tags, the input prompt text for
    # tokenization should contain the entity-type specified tags (e.g. `<lead>` and
    # `<position>`). The entities will be extracted and converted to the specified tags.
    # The unexpected entities will be treated as unknown type.
    text = root.text or ""
    for element in root:
        entity_type = element.attrib["type"]
        start_token = entity_start_tokens.get(entity_type, entity_start_tokens[None])
        end_token = entity_end_tokens.get(entity_type, entity_end_tokens[None])
        text += start_token + (element.text or "") + end_token + (element.tail or "")

    if len(root) > 0 and "label" in root[0].attrib:
        return text, names, [element.attrib["label"] for element in root]
    return text, names, None


def create_aligned_labels(
    input_ids: list[int],
    entity_start_ids: list[int],
    probs: Optional[list[list[float]]] = None,
    labels: Optional[list[str]] = None,
    label_names: Optional[list[str]] = None,
) -> list[list[float]]:
    aligned_labels = []
    for token_id in input_ids:
        if token_id not in entity_start_ids:
            aligned_labels.append([0.0, 0.0, 0.0])
            continue
        if probs is not None:
            aligned_labels.append(probs.pop(0))
            continue

        # Use one-hot label encoding with using `label_names`.
        label = label_names.index(labels.pop(0))
        label = [1.0 if i == label else 0.0 for i in range(len(label_names))]
        aligned_labels.append(label)
    return aligned_labels


def prepare_examples_with_tokenization(
    sources: list[str],
    tokenizer: PreTrainedTokenizerBase,
    entity_start_tokens: dict[Optional[str], str],
    entity_end_tokens: dict[Optional[str], str],
    newline_token: Optional[str] = None,
    label_names: Optional[list[str]] = None,
    **kwargs: Any,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    entity_names, encodings = [], []
    entity_start_ids = [tokenizer.vocab[x] for x in entity_start_tokens.values()]

    for source in sources:
        text, names, labels = convert_entities_from_xml(
            source=source,
            entity_start_tokens=entity_start_tokens,
            entity_end_tokens=entity_end_tokens,
        )
        entity_names.append(names)

        # Tokenize the input prompt text with replacing the line-break characters with
        # the given newline token and create aligned labels if the original ones exist.
        encoding = dict(tokenizer(text.replace("\n", newline_token or "\n"), **kwargs))
        if labels is not None:
            encoding["labels"] = create_aligned_labels(
                encoding["input_ids"],
                entity_start_ids,
                labels=labels,
                label_names=label_names,
            )
        encodings.append(encoding)
    return entity_names, encodings


class ChainDataCollator:
    def __init__(self, *collators: Callable):
        self.collators = collators

    def __call__(self, batch: list[Any]) -> Any:
        for collator in self.collators:
            batch = collator(batch)
        return batch


@dataclass
class DataCollatorWithEntityReplacement:
    entity_ids: list[int]
    replace_prob: Optional[float] = None
    replace_start_id: Optional[int] = None
    replace_end_id: Optional[int] = None
    random_state: Optional[np.random.RandomState] = None
    return_tensors: Optional[str] = None

    def _sample_replacement(self) -> bool:
        random_state = self.random_state or np.random
        return self.replace_prob and random_state.random() < self.replace_prob

    def _create_replaced_entities(self, input_ids: list[int]) -> list[int]:
        input_ids, replacement = input_ids.copy(), False
        for i, token_id in enumerate(input_ids):
            if token_id not in self.entity_ids:
                continue
            if not replacement and self._sample_replacement():
                input_ids[i], replacement = self.replace_start_id, True
            elif replacement:
                input_ids[i], replacement = self.replace_end_id, False
        return input_ids

    def __call__(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for i, example in enumerate(batch):
            batch[i] = example = example.copy()
            example["input_ids"] = self._create_replaced_entities(example["input_ids"])

        if self.return_tensors == "pt":
            return torch_default_data_collator(batch)
        return batch
