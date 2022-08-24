from __future__ import annotations

import argparse
import glob
import os
import warnings

import pandas as pd
import torch
import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from data import prepare_examples_with_tokenization

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


def prepare_inference_dataloader(
    args: argparse.Namespace,
) -> tuple[DataLoader, list[list[str]], list[int]]:
    sources, filenames = [], sorted(glob.glob(os.path.join(args.directory, "*.xml")))
    if args.validate:
        splits = KFold(5, shuffle=True, random_state=42).split(filenames)
        filenames = [filenames[i] for i in list(splits)[args.fold_index][1]]

    for filename in filenames:
        with open(filename) as fp:
            sources.append(fp.read())

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    entity_ids, encodings = prepare_examples_with_tokenization(
        sources=sources,
        tokenizer=tokenizer,
        entity_start_tokens=ENTITY_START_TOKENS,
        entity_end_tokens=ENTITY_END_TOKENS,
        newline_token=NEWLINE_TOKEN,
        label_names=LABEL_NAMES,
        truncation=True,
        max_length=args.max_length,
    )
    for encoding in encodings:
        if "labels" in encoding:
            encoding.pop("labels")

    sorted_indices = sorted(enumerate(encodings), key=lambda x: len(x[1]["input_ids"]))
    filenames = [filenames[i] for i, _ in sorted_indices]
    entity_ids = [entity_ids[i] for i, _ in sorted_indices]
    encodings = [encodings[i] for i, _ in sorted_indices]

    dataloader = DataLoader(
        encodings,
        batch_size=args.batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )
    entity_start_ids = [tokenizer.vocab[x] for x in ENTITY_START_TOKENS.values()]
    return dataloader, entity_ids, entity_start_ids


@torch.inference_mode()
def main(args: argparse.Namespace):
    dataloader, entity_ids, entity_start_ids = prepare_inference_dataloader(args)
    model = AutoModelForTokenClassification.from_pretrained(args.model).eval().cuda()

    outputs = []
    for batch in tqdm.tqdm(dataloader):
        batch = {k: v.cuda() for k, v in batch.items()}
        batch_probs = model(**batch).logits.softmax(dim=2).tolist()
        for input_ids, probs in zip(batch["input_ids"].tolist(), batch_probs):
            discourse_ids = entity_ids.pop(0)
            for token, prob in zip(input_ids, probs):
                if token not in entity_start_ids:
                    continue
                prob = dict(zip(LABEL_NAMES, prob))
                outputs.append({"discourse_id": discourse_ids.pop(0), **prob})

    pd.DataFrame(outputs).to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--directory", default="resources/train_xml")
    parser.add_argument("--validate", action="store_true", default=False)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1400)
    main(parser.parse_args())
