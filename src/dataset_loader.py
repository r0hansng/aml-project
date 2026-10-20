"""Utilities for loading and validating Universal Dependencies CoNLL-U corpora."""

import re

UPOS_TAGS = frozenset({
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
})

INTEGER_ID_PATTERN = re.compile(r"^\d+$")
MULTIWORD_ID_PATTERN = re.compile(r"^\d+-\d+$")
EMPTY_NODE_ID_PATTERN = re.compile(r"^\d+\.\d+$")


def load_conllu(path):
    """Load word and UPOS tag sequences from a CoNLL-U format file."""
    sentences = []
    current_sentence = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            if line.startswith("#"):
                continue

            columns = line.split("\t")
            if len(columns) != 10:
                raise ValueError(f"Invalid CoNLL-U line: {line}")

            token_id, word, pos_tag = columns[0], columns[1], columns[3]

            # Skip multiword tokens and empty nodes
            if MULTIWORD_ID_PATTERN.fullmatch(token_id) or EMPTY_NODE_ID_PATTERN.fullmatch(token_id):
                continue
            if not INTEGER_ID_PATTERN.fullmatch(token_id):
                raise ValueError(f"Invalid token ID: {token_id}")

            current_sentence.append((word, pos_tag))

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def extract_words_tags(sentences):
    """Split sentence token pairs into parallel word and tag sequences."""
    return [[w for w, _ in s] for s in sentences], [[t for _, t in s] for s in sentences]


def get_tokens(sentences):
    """Flatten sentence-level token pairs."""
    return [token for sentence in sentences for token in sentence]


def validate_sentences(sentences):
    """Ensure sentences and token structures are well-formed."""
    if not sentences:
        raise ValueError("No sentences were loaded")

    for i, sentence in enumerate(sentences):
        if not sentence:
            raise ValueError(f"Empty sentence at index {i}")
        for j, token in enumerate(sentence):
            if len(token) != 2 or not token[0] or not token[1]:
                raise ValueError(f"Invalid token at sentence {i}, token {j}: {token}")


def validate_upos_tags(sentences, allowed_tags=UPOS_TAGS):
    """Ensure all tags adhere to canonical UPOS inventory."""
    for i, sentence in enumerate(sentences):
        for j, (_, tag) in enumerate(sentence):
            if tag not in allowed_tags:
                raise ValueError(f"Invalid UPOS tag '{tag}' at sentence {i}, token {j}")