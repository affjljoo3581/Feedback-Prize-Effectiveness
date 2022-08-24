from __future__ import annotations

import argparse
import glob
import os
import xml.etree.ElementTree as ET
from collections import Counter

import pandas as pd
import tqdm

DISCOURSE_TYPES = [
    "Lead",
    "Position",
    "Claim",
    "Counterclaim",
    "Rebuttal",
    "Evidence",
    "Concluding Statement",
]


def create_metadata_from_xml(name: str, source: str) -> pd.DataFrame:
    # Parse XML by wrapping the source with `<root>` tag and extract full essay text.
    root = ET.fromstringlist(["<root>", source, "</root>"])
    essay = "".join(root.itertext())

    metadata, previous = [], root.text or ""
    for element in root:
        discourse_text = element.text or ""
        row = {
            "discourse_id": element.attrib["name"],
            "discourse_length": len(discourse_text),
            "discourse_portion": len(discourse_text) / (len(essay) or 1),
            "num_discourse_words": len(discourse_text.split()),
            "num_discourse_paragraphs": discourse_text.count("\n"),
            "discourse_start_position": len(previous),
            "discourse_start_ratio": len(previous) / (len(essay) or 1),
            "discourse_start_word_offset": len(previous.split()),
            "discourse_start_paragraph": previous.count("\n"),
            "discourse_start_paragraph_ratio": (
                previous.count("\n") / (essay.count("\n") or 1)
            ),
            "discourse_end_position": len(previous + discourse_text),
            "discourse_end_ratio": len(previous + discourse_text) / (len(essay) or 1),
            "discourse_end_word_offset": len((previous + discourse_text).split()),
            "discourse_line_position": len(previous[max(previous.rfind("\n"), 0) :]),
            "discourse_line_word_offset": len(
                previous[max(previous.rfind("\n"), 0) :].split()
            ),
            "discourse_type": element.attrib["type"],
        }

        # If `label` exists, then it will be included as `discourse_effectiveness`.
        if "label" in element.attrib:
            row["discourse_effectiveness"] = element.attrib["label"]
        previous += discourse_text + (element.tail or "")
        metadata.append(row)

    # Create metadata dataframe and insert global metadata information (e.g. length of
    # the essay, number of words in the essay and etc.).
    metadata = pd.DataFrame(metadata).set_index("discourse_id", drop=True)
    metadata["essay_id"] = name
    metadata["essay_length"] = len(essay)
    metadata["num_essay_words"] = len(essay.split())
    metadata["num_essay_paragraphs"] = essay.count("\n")
    metadata["num_discourses_in_essay"] = len(root)

    # In addition, we will specify the statistics of some specific values.
    for column in [
        "discourse_length",
        "discourse_portion",
        "num_discourse_words",
        "num_discourse_paragraphs",
    ]:
        metadata[f"mean_{column}"] = metadata[column].mean()
        metadata[f"std_{column}"] = metadata[column].std() or 0.0

    discourse_type_counter = Counter(metadata.discourse_type)
    for discourse in DISCOURSE_TYPES:
        metadata[f"num_{discourse}_essay_has"] = discourse_type_counter[discourse]
    return metadata


def create_metadata_from_dataset(
    name: str, source: str, dataset: pd.DataFrame
) -> pd.DataFrame:
    metadata = []
    for example in dataset.itertuples():
        discourse_text = example.discourse_text or ""
        previous = source[: source.find(discourse_text.strip())]
        row = {
            "discourse_id": example.discourse_id,
            "discourse_length": len(discourse_text),
            "discourse_portion": len(discourse_text) / (len(source) or 1),
            "num_discourse_words": len(discourse_text.split()),
            "num_discourse_paragraphs": discourse_text.count("\n"),
            "discourse_start_position": len(previous),
            "discourse_start_ratio": len(previous) / (len(source) or 1),
            "discourse_start_word_offset": (len(previous.split())),
            "discourse_start_paragraph": previous.count("\n"),
            "discourse_start_paragraph_ratio": (
                previous.count("\n") / (source.count("\n") or 1)
            ),
            "discourse_end_position": len(previous + discourse_text),
            "discourse_end_ratio": len(previous + discourse_text) / (len(source) or 1),
            "discourse_end_word_offset": len((previous + discourse_text).split()),
            "discourse_line_position": len(previous[max(previous.rfind("\n"), 0) :]),
            "discourse_line_word_offset": len(
                previous[max(previous.rfind("\n"), 0) :].split()
            ),
            "discourse_type": example.discourse_type,
        }

        # If `label` exists, then it will be included as `discourse_effectiveness`.
        if hasattr(example, "discourse_effectiveness"):
            row["discourse_effectiveness"] = example.discourse_effectiveness
        metadata.append(row)

    # Create metadata dataframe and insert global metadata information (e.g. length of
    # the essay, number of words in the essay and etc.).
    metadata = pd.DataFrame(metadata).set_index("discourse_id", drop=True)
    metadata["essay_id"] = name
    metadata["essay_length"] = len(source)
    metadata["num_essay_words"] = len(source.split())
    metadata["num_essay_paragraphs"] = source.count("\n")
    metadata["num_discourses_in_essay"] = len(source)

    # In addition, we will specify the statistics of some specific values.
    for column in [
        "discourse_length",
        "discourse_portion",
        "num_discourse_words",
        "num_discourse_paragraphs",
    ]:
        metadata[f"mean_{column}"] = metadata[column].mean()
        metadata[f"std_{column}"] = metadata[column].std() or 0.0

    discourse_type_counter = Counter(metadata.discourse_type)
    for discourse in DISCOURSE_TYPES:
        metadata[f"num_{discourse}_essay_has"] = discourse_type_counter[discourse]
    return metadata


def main(args: argparse.Namespace):
    # Concatenate the predicted probabilities from various models.
    predictions = []
    for filename in args.filenames:
        prefix = os.path.splitext(os.path.basename(filename))[0] + "-"
        prediction = pd.read_csv(filename, index_col="discourse_id").add_prefix(prefix)
        predictions.append(prediction)
    predictions = pd.concat(predictions, axis=1) if predictions else None

    # Generate metadata for the essays and their discourse. The metadata for nonexistent
    # discourses will be dropped from the predictions.
    metadata = []
    if args.dataset is None:
        for filename in tqdm.tqdm(glob.glob(os.path.join(args.xml_dir, "*.xml"))):
            with open(filename) as fp:
                essay_id = os.path.splitext(os.path.basename(filename))[0]
                metadata.append(create_metadata_from_xml(essay_id, fp.read()))
    else:
        dataset = pd.read_csv(args.dataset)
        for essay_id, data in tqdm.tqdm(dataset.groupby("essay_id")):
            with open(os.path.join(args.text_dir, f"{essay_id}.txt")) as fp:
                metadata.append(create_metadata_from_dataset(essay_id, fp.read(), data))
    metadata = pd.concat(metadata)

    if predictions is not None:
        metadata = pd.merge(metadata, predictions, left_index=True, right_index=True)
    metadata.to_csv(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    parser.add_argument("--dataset")
    parser.add_argument("--text-dir", default="resources/train")
    parser.add_argument("--xml-dir", default="resources/train_xml")
    parser.add_argument("--output", default="metadata.csv")
    main(parser.parse_args())
