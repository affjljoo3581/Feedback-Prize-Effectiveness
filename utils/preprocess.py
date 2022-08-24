from __future__ import annotations

import argparse
import codecs
import os
import re
import string
from xml.sax.saxutils import escape as escape_xml

import pandas as pd
import tqdm
from text_unidecode import unidecode

UNACCEPTABLE_CHARS = string.punctuation + string.whitespace


def replace_encoding_with_utf8(error: UnicodeError) -> tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end


# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    text = text.encode("raw_unicode_escape")
    text = text.decode("utf-8", errors="replace_decoding_with_cp1252")
    text = text.encode("cp1252", errors="replace_encoding_with_utf8")
    text = text.decode("utf-8", errors="replace_decoding_with_cp1252")
    return unidecode(text)


def find_discourse_from_essay(essay: str, discourse: str) -> int:
    # We observed that some discourse texts are not exactly same of the essay because of
    # the mismatch of data anonymization. Because of that, we will find the offset by
    # truncating from right to the left until they are matched.
    offset, endpoints = -1, [match.end() for match in re.finditer(r"\S+", discourse)]
    while offset < 0 and endpoints:
        offset = essay.find(discourse[: endpoints.pop()])
    return offset


def insert_discourse_tags_into_essay(essay: str, entities: pd.DataFrame) -> str:
    # Sort discources by their lengths. We will match the longer sentences first.
    entities["discourse_length"] = entities.discourse_text.str.len()
    entities = entities.sort_values("discourse_length", ascending=False)

    discourses, masked_essay, content = [], essay, ""
    word_spans = [match.span() for match in re.finditer(r"\S+", essay)]

    for entity in entities.itertuples():
        text = entity.discourse_text.lstrip(UNACCEPTABLE_CHARS).rstrip()
        words = text.split()

        # Find the start offset of the discourse with fixing the word truncation.
        start = find_discourse_from_essay(masked_essay, text)
        while start > 0 and masked_essay[start - 1] not in UNACCEPTABLE_CHARS:
            start -= 1

        # Find the word index from the word spans and get the offset range of the last
        # word in the discourse text.
        for word_index, (i, j) in enumerate(word_spans):
            if i <= start < j:
                break
        word_start, word_end = word_spans[word_index + len(words) - 1]

        # Calculate the endpoint of the discourse text with fixing the word truncation
        # and word overflow. It means that sometimes the overlapping between discourses
        # are happend and therefore we will use masked essay content to resize the
        # discourse range not to be overlapped with other discourses.
        end = word_start + min(word_end - word_start, len(words[-1]))
        while end > word_start and masked_essay[end - 1] in string.whitespace:
            end -= 1

        masked_essay = masked_essay[:start] + " " * (end - start) + masked_essay[end:]
        discourses.append((entity, start, end))

    # Sort the discourse spans by their start position.
    discourses = [(None, 0, 0)] + sorted(discourses, key=lambda item: item[1])
    for i, (discourse, start, end) in enumerate(discourses[1:]):
        # Create attributes of the discourse example. Note that the discourse
        # effectiveness only exists on the train dataset, so the label attribute is
        # optional.
        attribute = f'name="{discourse.discourse_id}" type="{discourse.discourse_type}"'
        if "discourse_effectiveness" in dir(discourse):
            attribute += f' label="{discourse.discourse_effectiveness}"'

        # Add the normal text and discourse element with escaping for XML format.
        content += escape_xml(essay[discourses[i][2] : start])
        content += f"<entity {attribute}>{escape_xml(essay[start:end])}</entity>"
    content += escape_xml(essay[discourses[-1][2] :])

    # Remove the duplicated whitespaces and change to the single whitespace.
    while "  " in content:
        content = content.replace("  ", " ")
    return content


def main(args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)

    # Normalize the discourse texts because the essay text will be normalized as well.
    # And there should be improper discourse types, so we will fill with empty space.
    data = pd.read_csv(args.entities)
    data.discourse_text = data.discourse_text.apply(resolve_encodings_and_normalize)
    data.discourse_type = data.discourse_type.fillna("Unknown")

    if "essay_id" not in data and "id" in data:
        data["essay_id"] = data.id
    if data.discourse_id.dtype == float or data.discourse_id.dtype == int:
        # Change the `discourse_id` which have float values to hex-format for handling
        # the column as string type.
        data.discourse_id = data.discourse_id.astype(int)
        data.discourse_id = data.discourse_id.apply(lambda x: format(x, "X"))

    for essay_id, entities in tqdm.tqdm(data.groupby("essay_id")):
        with open(os.path.join(args.text_dir, f"{essay_id}.txt")) as fp:
            essay = resolve_encodings_and_normalize(fp.read())
        with open(os.path.join(args.output_dir, f"{essay_id}.xml"), "w") as fp:
            fp.write(insert_discourse_tags_into_essay(essay, entities))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-dir", default="resources/train")
    parser.add_argument("--output-dir", default="resources/train_xml")
    parser.add_argument("--entities", default="resources/train.csv")
    main(parser.parse_args())
