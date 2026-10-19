"""Tests for official dataset registration and CoNLL-U edge cases."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_loader import (
    UPOS_TAGS,
    load_conllu,
    validate_sentences,
    validate_upos_tags,
)
from src.dataset_registry import DATASET_SPLITS, validate_dataset_files


EXPECTED_SENTENCE_COUNTS = {
    "train": 12544,
    "dev": 2001,
    "test": 2077,
}

EXPECTED_TOKEN_COUNTS = {
    "train": 204578,
    "dev": 25148,
    "test": 25094,
}


class TestOfficialDatasetSplits(unittest.TestCase):
    """Verify the registered EWT files and their expected inventory."""

    def test_official_split_files_exist(self):
        """All official train, dev, and test files must be available."""
        validate_dataset_files()

        for split_name, split_path in DATASET_SPLITS.items():
            with self.subTest(split=split_name):
                self.assertTrue(split_path.is_file())

    def test_official_sentence_counts(self):
        """The official split sentence counts must remain unchanged."""
        for split_name, split_path in DATASET_SPLITS.items():
            sentences = load_conllu(split_path)

            with self.subTest(split=split_name):
                self.assertEqual(
                    len(sentences),
                    EXPECTED_SENTENCE_COUNTS[split_name],
                )

    def test_official_token_counts(self):
        """The official split token counts must remain unchanged."""
        for split_name, split_path in DATASET_SPLITS.items():
            sentences = load_conllu(split_path)
            token_count = sum(len(sentence) for sentence in sentences)

            with self.subTest(split=split_name):
                self.assertEqual(
                    token_count,
                    EXPECTED_TOKEN_COUNTS[split_name],
                )

    def test_official_sentences_use_valid_upos_tags(self):
        """All official tokens must use the defined UPOS tag set."""
        for split_name, split_path in DATASET_SPLITS.items():
            sentences = load_conllu(split_path)
            validate_sentences(sentences)
            validate_upos_tags(sentences, UPOS_TAGS)

            with self.subTest(split=split_name):
                self.assertTrue(sentences)


class TestConlluEdgeCases(unittest.TestCase):
    """Verify the loader's handling of valid and invalid CoNLL-U rows."""

    def load_temporary_conllu(self, content):
        """Load temporary CoNLL-U content and remove the temporary file."""
        with TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "sample.conllu"
            file_path.write_text(content, encoding="utf-8")
            return load_conllu(file_path)

    def test_skips_comments_multiword_tokens_and_empty_nodes(self):
        """Comments and non-word rows must not enter the token output."""
        content = (
            "# sent_id = example\r\n"
            "1-2\tcan't\t_\t_\t_\t_\t_\t_\t_\t_\r\n"
            "1\tcan\t_\tAUX\t_\t_\t0\troot\t_\t_\r\n"
            "2.1\tanalyzed\t_\tVERB\t_\t_\t1\tdep\t_\t_\r\n"
            "2\tnot\t_\tPART\t_\t_\t1\tadvmod\t_\t_\r\n"
            "\r\n"
        )

        sentences = self.load_temporary_conllu(content)

        self.assertEqual(
            sentences,
            [[("can", "AUX"), ("not", "PART")]],
        )

    def test_rejects_invalid_column_count(self):
        """A CoNLL-U row must contain exactly ten columns."""
        content = "1\tword\t_\tNOUN\n"

        with self.assertRaises(ValueError):
            self.load_temporary_conllu(content)

    def test_rejects_invalid_token_id(self):
        """A token row must use an integer, multiword, or empty-node ID."""
        content = "abc\tword\t_\tNOUN\t_\t_\t0\troot\t_\t_\n"

        with self.assertRaises(ValueError):
            self.load_temporary_conllu(content)

    def test_rejects_unknown_upos_tag(self):
        """A tag outside the official UPOS set must be rejected."""
        sentences = [[("word", "INVALID")]]

        with self.assertRaises(ValueError):
            validate_upos_tags(sentences)

    def test_rejects_empty_sentence_collection(self):
        """Validation must reject a dataset with no sentences."""
        with self.assertRaises(ValueError):
            validate_sentences([])


if __name__ == "__main__":
    unittest.main()
