"""First-Order Hidden Markov Model (HMM) for Part-of-Speech Tagging."""

from collections import Counter, defaultdict
from math import log

from src.preprocessing import (
    END_TAG,
    START_TAG,
    UNKNOWN_WORD,
    HMMVocabulary,
    build_vocabulary,
    normalize_word,
)


class HMMPOSTagger:
    """First-order HMM tagger with add-alpha smoothing and log-space Viterbi decoding."""

    def __init__(
        self,
        smoothing: float = 0.1,
        minimum_word_frequency: int = 1,
    ):
        if smoothing <= 0:
            raise ValueError("smoothing must be greater than zero")
        if minimum_word_frequency < 1:
            raise ValueError("minimum_word_frequency must be at least one")

        self.smoothing = smoothing
        self.minimum_word_frequency = minimum_word_frequency
        self.vocabulary: HMMVocabulary | None = None
        self.transition_log_probabilities: dict[str, dict[str, float]] = {}
        self.emission_log_probabilities: dict[str, dict[str, float]] = {}

    @property
    def original_vocabulary(self) -> frozenset[str]:
        """Empirical training vocabulary, excluding the UNK placeholder."""
        if self.vocabulary is None:
            raise RuntimeError("Model must be fitted before accessing original_vocabulary")
        return self.vocabulary.original_words

    def fit(self, sentences: list[list[tuple[str, str]]]):
        """Estimate transition and emission log-probabilities from annotated sentences."""
        if not sentences or not any(sentences):
            raise ValueError("Training data must contain at least one non-empty sentence")

        self.vocabulary = build_vocabulary(
            sentences,
            minimum_word_frequency=self.minimum_word_frequency,
        )

        trans_counts, emit_counts = self._accumulate_counts(sentences)
        self.transition_log_probabilities = self._smooth_transitions(trans_counts)
        self.emission_log_probabilities = self._smooth_emissions(emit_counts)
        return self

    def predict(self, words: list[str]) -> list[str]:
        """Decode the optimal tag sequence for a sentence via log-space Viterbi."""
        if self.vocabulary is None:
            raise RuntimeError("The model must be fitted before prediction")
        if not words:
            return []

        return self._viterbi_decode(words)

    def predict_batch(self, sentences: list[list[str]]) -> list[list[str]]:
        """Decode optimal tag sequences for a batch of tokenized sentences."""
        return [self.predict(sentence) for sentence in sentences]

    def _accumulate_counts(self, sentences):
        trans_counts = defaultdict(Counter)
        emit_counts = defaultdict(Counter)

        for sentence in sentences:
            if not sentence:
                continue

            prev_tag = START_TAG
            for word, tag in sentence:
                trans_counts[prev_tag][tag] += 1
                norm_word = normalize_word(word, self.vocabulary)
                emit_counts[tag][norm_word] += 1
                prev_tag = tag

            trans_counts[prev_tag][END_TAG] += 1

        return trans_counts, emit_counts

    def _smooth_transitions(self, trans_counts):
        tags = list(self.vocabulary.tags)
        log_transitions = {}

        for prev_tag in [START_TAG, *tags]:
            next_tags = tags if prev_tag == START_TAG else [*tags, END_TAG]
            total = sum(trans_counts[prev_tag].values())
            denom = total + self.smoothing * len(next_tags)

            log_transitions[prev_tag] = {
                next_tag: log((trans_counts[prev_tag][next_tag] + self.smoothing) / denom)
                for next_tag in next_tags
            }

        return log_transitions

    def _smooth_emissions(self, emit_counts):
        vocab_words = sorted(self.vocabulary.words)
        vocab_size = len(vocab_words)
        log_emissions = {}

        for tag in self.vocabulary.tags:
            total = sum(emit_counts[tag].values())
            denom = total + self.smoothing * vocab_size

            log_emissions[tag] = {
                word: log((emit_counts[tag][word] + self.smoothing) / denom)
                for word in vocab_words
            }

        return log_emissions

    def _viterbi_decode(self, words: list[str]) -> list[str]:
        tags = self.vocabulary.tags
        first_word = normalize_word(words[0], self.vocabulary)

        # Base case: t = 1 (transition from START)
        viterbi = {
            tag: (
                self.transition_log_probabilities[START_TAG][tag]
                + self.emission_log_probabilities[tag][first_word]
            )
            for tag in tags
        }
        backpointers: list[dict[str, str]] = []

        # Induction: t > 1
        for word in words[1:]:
            norm_word = normalize_word(word, self.vocabulary)
            next_viterbi = {}
            step_backpointers = {}

            for curr_tag in tags:
                emit_score = self.emission_log_probabilities[curr_tag][norm_word]
                best_prev_tag, best_score = max(
                    (
                        (prev_tag, viterbi[prev_tag] + self.transition_log_probabilities[prev_tag][curr_tag])
                        for prev_tag in tags
                    ),
                    key=lambda x: x[1],
                )
                next_viterbi[curr_tag] = best_score + emit_score
                step_backpointers[curr_tag] = best_prev_tag

            viterbi = next_viterbi
            backpointers.append(step_backpointers)

        # Termination: transition to END and backtrace
        best_last_tag, _ = max(
            (
                (tag, viterbi[tag] + self.transition_log_probabilities[tag][END_TAG])
                for tag in tags
            ),
            key=lambda x: x[1],
        )

        path = [best_last_tag]
        for step_bp in reversed(backpointers):
            path.append(step_bp[path[-1]])

        return path[::-1]