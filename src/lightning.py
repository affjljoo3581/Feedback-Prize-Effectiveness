from __future__ import annotations

import glob
import os
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from sklearn.model_selection import KFold
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DebertaConfig,
    DebertaV2Config,
    PreTrainedTokenizerBase,
    get_scheduler,
)

from data import (
    ChainDataCollator,
    DataCollatorWithEntityReplacement,
    create_aligned_labels,
    prepare_examples_with_tokenization,
)

try:
    from apex.optimizers import FusedAdam as AdamW
except ModuleNotFoundError:
    from torch.optim import AdamW

ENTITY_START_TOKENS = {
    "Lead": "<lead>",
    "Position": "<position>",
    "Claim": "<claim>",
    "Counterclaim": "<counterclaim>",
    "Rebuttal": "<rebuttal>",
    "Evidence": "<evidence>",
    "Concluding Statement": "<concludingstatement>",
    None: "<unknown>",
}
ENTITY_END_TOKENS = {
    "Lead": "</lead>",
    "Position": "</position>",
    "Claim": "</claim>",
    "Counterclaim": "</counterclaim>",
    "Rebuttal": "</rebuttal>",
    "Evidence": "</evidence>",
    "Concluding Statement": "</concludingstatement>",
    None: "</unknown>",
}
NEWLINE_TOKEN = "<br>"
LABEL_NAMES = ["Ineffective", "Adequate", "Effective"]


class MyLightningModule(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config = config.copy()
        self.evaluate_start_step = config.train.evaluate_start_step

        # Fix the random seed and create transformer model for token classification.
        torch.manual_seed(config.model.random_seed)
        self.model = AutoModelForTokenClassification.from_pretrained(
            **config.model.transformer, num_labels=len(LABEL_NAMES)
        )

        # If the absolute-position attention transformer model has smaller position
        # embeddings, then we will expand to the corresponding size and overwrite with
        # the original position embeddings.
        if (
            not isinstance(self.model.config, (DebertaV2Config, DebertaConfig))
            and self.model.config.max_position_embeddings < config.data.max_length
        ):
            model_config = AutoConfig.from_pretrained(
                **config.model.transformer,
                max_position_embeddings=config.data.max_length + 2,
                num_labels=len(LABEL_NAMES),
            )
            model = AutoModelForTokenClassification.from_config(model_config)

            state_dict = model.state_dict()
            for k, v in self.model.state_dict().items():
                if v.shape == state_dict[k].shape:
                    state_dict[k] = v
                elif "position_embeddings" in k:
                    state_dict[k][: v.shape[0]] = v
            model.load_state_dict(state_dict)
            self.model = model

        # Add new embeddings for the special additional tokens.
        entity_tokens = len(ENTITY_START_TOKENS) + len(ENTITY_END_TOKENS) + 1
        self.model.resize_token_embeddings(self.model.config.vocab_size + entity_tokens)

        # Enable the gradient checkpointing to reduce the memory usage if specified.
        if config.train.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(self, **batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        labels = batch.pop("labels").float()
        labels_mask = labels.sum(2) > 0

        logits = self.model(**batch).logits.float()
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            labels.flatten(0, 1),
            reduction="sum",
        )
        loss = loss / labels_mask.sum()

        accuracy = logits.argmax(dim=2) == labels.argmax(dim=2)
        accuracy = (labels_mask * accuracy).sum() / labels_mask.sum()
        return loss, accuracy

    def training_step(self, batch: dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        loss, accuracy = self(**batch)
        self.log("step", self.global_step)
        self.log("train/loss", loss)
        self.log("train/accuracy", accuracy)
        return loss

    def training_step_end(self, outputs: list[torch.Tensor]):
        if self.trainer.limit_val_batches != 0.0:
            return
        if self.global_step > self.evaluate_start_step:
            self.trainer.limit_val_batches = 1.0
            self.trainer.reset_val_dataloader()

    def validation_step(self, batch: dict[str, torch.Tensor], idx: int):
        loss, accuracy = self(**batch)
        self.log("step", self.global_step)
        self.log("val/loss", loss)
        self.log("val/accuracy", accuracy)

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        do_decay = [p for p in self.parameters() if p.ndim >= 2]
        no_decay = [p for p in self.parameters() if p.ndim < 2]
        param_groups = [{"params": do_decay}, {"params": no_decay, "weight_decay": 0.0}]

        optimizer = AdamW(param_groups, **self.config.optim.optimizer)
        scheduler = get_scheduler(optimizer=optimizer, **self.config.optim.scheduler)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class MyLightningDataModule(LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def load_external_dataset(self, tokenizer: PreTrainedTokenizerBase):
        sources = []
        pattern = os.path.join(self.config.data.external_directory, "*.xml")

        for filename in sorted(glob.glob(pattern)):
            with open(filename) as fp:
                sources.append(fp.read())

        # Create external dataset examples and tokenize them with padding and
        # truncation.
        entity_names, encodings = prepare_examples_with_tokenization(
            sources=sources,
            tokenizer=self.tokenizer,
            entity_start_tokens=ENTITY_START_TOKENS,
            entity_end_tokens=ENTITY_END_TOKENS,
            newline_token=NEWLINE_TOKEN,
            label_names=LABEL_NAMES,
            truncation=True,
            max_length=self.config.data.max_length,
        )
        entity_start_ids = [tokenizer.vocab[x] for x in ENTITY_START_TOKENS.values()]

        # Read the pseudo-labels and create aligned labels which consist of the
        # probability of each category.
        labels = pd.read_csv(self.config.data.external_labels, index_col="discourse_id")
        labels = labels.to_dict(orient="index")
        labels = defaultdict(lambda: {label: 1 / 3 for label in LABEL_NAMES}, labels)

        for names, encoding in zip(entity_names, encodings):
            encoding["labels"] = create_aligned_labels(
                encoding["input_ids"],
                entity_start_ids,
                probs=[[labels[name][cat] for cat in LABEL_NAMES] for name in names],
            )
        return encodings

    def setup(self, stage: Optional[str] = None):
        # Create a tokenizer with additional special tokens.
        self.tokenizer = AutoTokenizer.from_pretrained(**self.config.model.transformer)
        self.tokenizer.add_tokens(list(ENTITY_START_TOKENS.values()))
        self.tokenizer.add_tokens(list(ENTITY_END_TOKENS.values()))
        self.tokenizer.add_tokens([NEWLINE_TOKEN])

        # Load the dataset by reading all XML files in the given directory path.
        sources, pattern = [], os.path.join(self.config.data.directory, "*.xml")
        for filename in sorted(glob.glob(pattern)):
            with open(filename) as fp:
                sources.append(fp.read())

        # Create K-Fold splitter and create shuffled train indices and validation
        # indices according to the given number-of-folds and the target fold index.
        fold = KFold(self.config.data.num_folds, shuffle=True, random_state=42)
        train_idx, val_idx = list(fold.split(sources))[self.config.data.fold_index]

        # Create dataset examples for training and tokenize them with padding and
        # truncation. And then we will split them into train and validation encodings by
        # using their indices computed above.
        _, encodings = prepare_examples_with_tokenization(
            sources=sources,
            tokenizer=self.tokenizer,
            entity_start_tokens=ENTITY_START_TOKENS,
            entity_end_tokens=ENTITY_END_TOKENS,
            newline_token=NEWLINE_TOKEN,
            label_names=LABEL_NAMES,
            truncation=True,
            max_length=self.config.data.max_length,
        )
        self.train_dataset = [encodings[i] for i in train_idx]
        self.val_dataset = sorted(
            [encodings[i] for i in val_idx],
            key=lambda encoding: len(encoding["input_ids"]),
        )

        # Load the external dataset with their pseudo-labels and add to the train
        # encodings. After that, the train examples will be shuffled according to the
        # given random seed.
        if "external_directory" in self.config.data:
            self.train_dataset += self.load_external_dataset(self.tokenizer)
        np.random.RandomState(self.config.data.random_seed).shuffle(self.train_dataset)

        # Extract the entity token ids and create a data collator for random entity
        # replacement. The data collator will replace the original entities with the
        # given unknown tokens with probability of `replace_prob`. Note that the random
        # replacement should be disabled on validation, so `DataCollatorForSeq2Seq` will
        # be used only.
        entity_ids = []
        entity_ids.extend(self.tokenizer.vocab[x] for x in ENTITY_START_TOKENS.values())
        entity_ids.extend(self.tokenizer.vocab[x] for x in ENTITY_END_TOKENS.values())

        self.augment_collator = DataCollatorWithEntityReplacement(
            entity_ids=entity_ids,
            replace_prob=self.config.data.replace_prob,
            replace_start_id=self.tokenizer.vocab[ENTITY_START_TOKENS[None]],
            replace_end_id=self.tokenizer.vocab[ENTITY_END_TOKENS[None]],
            random_state=np.random.RandomState(self.config.data.random_seed),
        )
        self.padding_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            max_length=self.config.data.max_length,
            pad_to_multiple_of=8,
            label_pad_token_id=[0.0, 0.0, 0.0],
            return_tensors="pt",
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            collate_fn=ChainDataCollator(self.augment_collator, self.padding_collator),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.train.batch_size,
            collate_fn=self.padding_collator,
        )
