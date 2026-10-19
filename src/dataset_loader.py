"""Utilities for loading and validating Universal Dependencies data."""
import re

UPOS_TAGS = frozenset(
    {
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
    }
)

INTEGER_ID_PATTERN = re.compile(r"^\d+$")
MULTIWORD_ID_PATTERN = re.compile(r"^\d+-\d+$")
EMPTY_NODE_ID_PATTERN = re.compile(r"^\d+\.\d+$")


def load_conllu(path):
    """Load word and universal POS-tag pairs from a CoNLL-U file.

    Comments and multiword or empty-node entries are ignored. Sentences are
    separated by blank lines and returned as lists of ``(word, tag)`` pairs.
    """
    sentences = []
    current_sentence = []

    with open(path, "r", encoding="utf-8") as data_file:
        for raw_line in data_file:
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

            token_id = columns[0]
            word = columns[1]
            pos_tag = columns[3]

            if MULTIWORD_ID_PATTERN.fullmatch(token_id):
                continue
            
            if EMPTY_NODE_ID_PATTERN.fullmatch(token_id):
                continue
            
            if not INTEGER_ID_PATTERN.fullmatch(token_id):
                raise ValueError(f"Invalid token ID: {token_id}")

            current_sentence.append((word, pos_tag))

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def extract_words_tags(sentences):
    """Split sentence token pairs into parallel word and tag lists."""
    words = []
    tags = []

    for token_sentence in sentences:
        sentence_words = []
        sentence_tags = []

        for word, pos_tag in token_sentence:
            sentence_words.append(word)
            sentence_tags.append(pos_tag)

        words.append(sentence_words)
        tags.append(sentence_tags)

    return words, tags


def get_tokens(sentences):
    """Flatten sentence-level token pairs into one list."""
    return [token for token_sentence in sentences for token in token_sentence]


def validate_sentences(sentences):
    """Raise ``ValueError`` if sentences contain invalid token data."""
    if not sentences:
        raise ValueError("No sentences were loaded")

    for sentence_index, token_sentence in enumerate(sentences):
        if not token_sentence:
            raise ValueError(
                f"Empty sentence at index {sentence_index}"
            )

        for token_index, token in enumerate(token_sentence):
            if len(token) != 2:
                raise ValueError(
                    f"Invalid token at sentence {sentence_index}, "
                    f"token {token_index}"
                )

            word, pos_tag = token

            if not word:
                raise ValueError(
                    f"Empty word at sentence {sentence_index}, "
                    f"token {token_index}"
                )

            if not pos_tag:
                raise ValueError(
                    f"Empty POS tag at sentence {sentence_index}, "
                    f"token {token_index}"
                )
                
def validate_upos_tags(sentences, allowed_tags=UPOS_TAGS):
    """Raise ValueError if a sentence contains an unknown UPOS tag."""
    for sentence_index, token_sentence in enumerate(sentences):
        for token_index, token in enumerate(token_sentence):
            _, pos_tag = token

            if pos_tag not in allowed_tags:
                raise ValueError(
                    f"Invalid UPOS tag '{pos_tag}' at "
                    f"sentence {sentence_index}, token {token_index}"
                )