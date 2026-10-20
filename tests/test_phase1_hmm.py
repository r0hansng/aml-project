"""Comprehensive unit tests for Phase 1 HMM POS tagger components."""

import unittest
from math import exp

from src.evaluation import evaluate_predictions, flatten_sequences
from src.hmm_model import HMMPOSTagger
from src.preprocessing import (
    END_TAG,
    START_TAG,
    UNKNOWN_WORD,
    HMMVocabulary,
    build_vocabulary,
    extract_original_vocabulary,
    normalize_word,
)


class TestPreprocessing(unittest.TestCase):
    """Test vocabulary building, normalization, and vocabulary separation."""

    def setUp(self):
        self.sample_sentences = [
            [("The", "DET"), ("dog", "NOUN"), ("barks", "VERB"), (".", "PUNCT")],
            [("The", "DET"), ("cat", "NOUN"), ("sleeps", "VERB"), (".", "PUNCT")],
            [("A", "DET"), ("dog", "NOUN"), ("runs", "VERB"), (".", "PUNCT")],
        ]

    def test_extract_original_vocabulary(self):
        """Original vocabulary contains distinct training words and excludes UNKNOWN_WORD."""
        vocab = extract_original_vocabulary(self.sample_sentences)
        expected = {"The", "dog", "barks", ".", "cat", "sleeps", "A", "runs"}
        self.assertEqual(vocab, expected)
        self.assertNotIn(UNKNOWN_WORD, vocab)

    def test_build_vocabulary_minimum_frequency_one(self):
        """build_vocabulary keeps words and original_words distinct."""
        vocab = build_vocabulary(self.sample_sentences, minimum_word_frequency=1)
        self.assertIsInstance(vocab, HMMVocabulary)

        # original_words must NOT include UNKNOWN_WORD
        self.assertNotIn(UNKNOWN_WORD, vocab.original_words)
        self.assertEqual(
            vocab.original_words,
            frozenset({"The", "dog", "barks", ".", "cat", "sleeps", "A", "runs"}),
        )

        # words must include UNKNOWN_WORD
        self.assertIn(UNKNOWN_WORD, vocab.words)
        self.assertEqual(len(vocab.words), len(vocab.original_words) + 1)

        # tags must be sorted tuple of unique tags
        self.assertEqual(vocab.tags, ("DET", "NOUN", "PUNCT", "VERB"))

    def test_build_vocabulary_minimum_frequency_two(self):
        """Low frequency words are pruned from model words but retained in original_words."""
        vocab = build_vocabulary(self.sample_sentences, minimum_word_frequency=2)

        # Words with frequency >= 2: The (2), dog (2), . (3)
        self.assertEqual(vocab.words, frozenset({"The", "dog", ".", UNKNOWN_WORD}))

        # original_words contains all original training words regardless of threshold
        self.assertIn("cat", vocab.original_words)
        self.assertIn("barks", vocab.original_words)
        self.assertNotIn(UNKNOWN_WORD, vocab.original_words)

    def test_normalize_word(self):
        """normalize_word returns the original word if known, else UNKNOWN_WORD."""
        vocab = build_vocabulary(self.sample_sentences, minimum_word_frequency=1)
        self.assertEqual(normalize_word("dog", vocab), "dog")
        self.assertEqual(normalize_word("elephant", vocab), UNKNOWN_WORD)


class TestHMMModel(unittest.TestCase):
    """Test HMMPOSTagger training, decoding, and probability properties."""

    def setUp(self):
        self.sample_sentences = [
            [("The", "DET"), ("dog", "NOUN"), ("barks", "VERB")],
            [("A", "DET"), ("cat", "NOUN"), ("meows", "VERB")],
            [("The", "DET"), ("cat", "NOUN"), ("barks", "VERB")],
        ]
        self.model = HMMPOSTagger(smoothing=0.1, minimum_word_frequency=1)

    def test_invalid_parameters(self):
        """Test parameter validation for smoothing and frequency."""
        with self.assertRaises(ValueError):
            HMMPOSTagger(smoothing=0)
        with self.assertRaises(ValueError):
            HMMPOSTagger(smoothing=-0.5)
        with self.assertRaises(ValueError):
            HMMPOSTagger(minimum_word_frequency=0)

    def test_empty_training_data(self):
        """Model raises ValueError on empty or all-empty training sentences."""
        with self.assertRaises(ValueError):
            self.model.fit([])
        with self.assertRaises(ValueError):
            self.model.fit([[], []])

    def test_predict_before_fit(self):
        """Predicting before fitting raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            self.model.predict(["The", "dog"])

    def test_original_vocabulary_property(self):
        """original_vocabulary property reflects training words without UNKNOWN_WORD."""
        with self.assertRaises(RuntimeError):
            _ = self.model.original_vocabulary

        self.model.fit(self.sample_sentences)
        self.assertEqual(
            self.model.original_vocabulary,
            frozenset({"The", "dog", "barks", "A", "cat", "meows"}),
        )
        self.assertNotIn(UNKNOWN_WORD, self.model.original_vocabulary)

    def test_transition_probabilities_sum_to_one(self):
        """Transition probabilities for each state sum to 1."""
        self.model.fit(self.sample_sentences)
        for prev_tag, trans in self.model.transition_log_probabilities.items():
            prob_sum = sum(exp(log_p) for log_p in trans.values())
            self.assertAlmostEqual(prob_sum, 1.0, places=5)

    def test_emission_probabilities_sum_to_one(self):
        """Emission probabilities for each tag sum to 1."""
        self.model.fit(self.sample_sentences)
        for tag, emissions in self.model.emission_log_probabilities.items():
            prob_sum = sum(exp(log_p) for log_p in emissions.values())
            self.assertAlmostEqual(prob_sum, 1.0, places=5)

    def test_predict_sequence_length(self):
        """Predicted tag sequence matches input word sequence length."""
        self.model.fit(self.sample_sentences)

        self.assertEqual(self.model.predict([]), [])
        tags_single = self.model.predict(["The"])
        self.assertEqual(len(tags_single), 1)

        tags_multi = self.model.predict(["The", "cat", "barks"])
        self.assertEqual(tags_multi, ["DET", "NOUN", "VERB"])

    def test_predict_batch(self):
        """predict_batch handles multiple sentences correctly."""
        self.model.fit(self.sample_sentences)
        batch = [["The", "dog"], ["A", "cat", "meows"]]
        predictions = self.model.predict_batch(batch)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(predictions[0]), 2)
        self.assertEqual(len(predictions[1]), 3)

    def test_unknown_word_prediction(self):
        """Model smoothly tags sentences with unknown words."""
        self.model.fit(self.sample_sentences)
        predictions = self.model.predict(["A", "giraffe", "sleeps"])
        self.assertEqual(len(predictions), 3)
        self.assertIn(predictions[0], self.model.vocabulary.tags)
        self.assertIn(predictions[1], self.model.vocabulary.tags)
        self.assertIn(predictions[2], self.model.vocabulary.tags)


class TestEvaluation(unittest.TestCase):
    """Test evaluation metrics and vocabulary handling."""

    def test_flatten_sequences(self):
        """flatten_sequences flattens nested lists."""
        nested = [["a", "b"], ["c"]]
        self.assertEqual(flatten_sequences(nested), ["a", "b", "c"])

    def test_evaluate_predictions_accuracy_and_unknown(self):
        """evaluate_predictions correctly computes overall and unknown-word metrics."""
        gold = [["DET", "NOUN"], ["VERB", "NOUN"]]
        pred = [["DET", "NOUN"], ["NOUN", "NOUN"]]
        words = [["The", "dog"], ["runs", "fast"]]
        train_vocab = {"The", "dog"}  # "runs" and "fast" are unknown

        metrics = evaluate_predictions(
            gold_sequences=gold,
            predicted_sequences=pred,
            word_sequences=words,
            training_vocabulary=train_vocab,
        )

        # 3 out of 4 correct overall: 0.75
        self.assertAlmostEqual(metrics["token_accuracy"], 0.75)
        # 2 unknown tokens: "runs" (gold VERB, pred NOUN - incorrect), "fast" (gold NOUN, pred NOUN - correct)
        self.assertEqual(metrics["unknown_word_count"], 2)
        self.assertAlmostEqual(metrics["unknown_word_accuracy"], 0.5)

        # Check structured per-tag metrics
        self.assertIn("per_tag_precision", metrics)
        self.assertIn("per_tag_recall", metrics)
        self.assertIn("per_tag_f1", metrics)
        self.assertIn("DET", metrics["per_tag_f1"])
        self.assertAlmostEqual(metrics["per_tag_f1"]["DET"], 1.0)

    def test_unk_excluded_from_training_vocab_check(self):
        """Even if <UNK> is erroneously in training_vocabulary, it is excluded."""
        gold = [["DET", "NOUN"]]
        pred = [["DET", "NOUN"]]
        words = [["<UNK>", "dog"]]
        # Pass a vocabulary that contains <UNK>
        train_vocab = {"dog", UNKNOWN_WORD}

        metrics = evaluate_predictions(
            gold_sequences=gold,
            predicted_sequences=pred,
            word_sequences=words,
            training_vocabulary=train_vocab,
        )

        # "<UNK>" token in the sentence should be recognized as an unknown word
        self.assertEqual(metrics["unknown_word_count"], 1)

    def test_mismatched_lengths_raise_error(self):
        """Sequence length mismatches raise ValueError."""
        with self.assertRaises(ValueError):
            evaluate_predictions(
                gold_sequences=[["DET"]],
                predicted_sequences=[["DET"], ["NOUN"]],
                word_sequences=[["The"]],
                training_vocabulary={"The"},
            )

    def test_empty_sequences_raise_error(self):
        """Empty sequences raise ValueError."""
        with self.assertRaises(ValueError):
            evaluate_predictions(
                gold_sequences=[],
                predicted_sequences=[],
                word_sequences=[],
                training_vocabulary=set(),
            )


if __name__ == "__main__":
    unittest.main()
