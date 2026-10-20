"""Corpus preprocessing and vocabulary utilities for HMM POS tagging."""

from collections import Counter
from dataclasses import dataclass

START_TAG = "<START>"
END_TAG = "<END>"
UNKNOWN_WORD = "<UNK>"


@dataclass(frozen=True)
class HMMVocabulary:
    """Container for model vocabulary, UPOS tagset, and empirical training lexicon."""

    words: frozenset[str]
    tags: tuple[str, ...]
    original_words: frozenset[str] = frozenset()


def extract_original_vocabulary(
    sentences: list[list[tuple[str, str]]],
) -> set[str]:
    """Extract distinct empirical word types from training sentences, excluding <UNK>."""
    return {
        word
        for sentence in sentences
        for word, _ in sentence
        if word != UNKNOWN_WORD
    }


def build_vocabulary(
    sentences: list[list[tuple[str, str]]],
    minimum_word_frequency: int = 1,
) -> HMMVocabulary:
    """Build model vocabulary with frequency pruning and preserve empirical training lexicon."""
    word_counts = Counter()
    tags = set()

    for sentence in sentences:
        for word, tag in sentence:
            word_counts[word] += 1
            tags.add(tag)

    original_words = frozenset(
        word for word in word_counts if word != UNKNOWN_WORD
    )
    known_words = {
        word
        for word, count in word_counts.items()
        if count >= minimum_word_frequency and word != UNKNOWN_WORD
    }
    known_words.add(UNKNOWN_WORD)

    return HMMVocabulary(
        words=frozenset(known_words),
        tags=tuple(sorted(tags)),
        original_words=original_words,
    )


def normalize_word(word: str, vocabulary: HMMVocabulary) -> str:
    """Map unseen word to UNK token."""
    return word if word in vocabulary.words else UNKNOWN_WORD