import tempfile
import unittest
from pathlib import Path

from src.dataset_loader import load_conllu


VALID_TOKEN = (
    "1\tword\t_\tNOUN\t_\t_\t0\troot\t_\t_"
)


class TestDatasetLoader(unittest.TestCase):
    def write_conllu(self, content):
        temporary_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".conllu",
            delete=False,
            encoding="utf-8",
        )
        temporary_file.write(content)
        temporary_file.close()
        return Path(temporary_file.name)

    def test_skips_comments_multiword_tokens_and_empty_nodes(self):
        content = (
            "# sent_id = example\n"
            "1-2\tcan't\t_\t_\t_\t_\t_\t_\t_\t_\n"
            "1\tcan\t_\tAUX\t_\t_\t0\troot\t_\t_\n"
            "2.1\tanalyzed\t_\tVERB\t_\t_\t1\tdep\t_\t_\n"
            "2\tnot\t_\tPART\t_\t_\t1\tadvmod\t_\t_\n"
            "\n"
        )

        file_path = self.write_conllu(content)

        try:
            sentences = load_conllu(file_path)
        finally:
            file_path.unlink()

        self.assertEqual(
            sentences,
            [[("can", "AUX"), ("not", "PART")]],
        )

    def test_rejects_invalid_column_count(self):
        file_path = self.write_conllu("1\tword\t_\tNOUN\n")

        try:
            with self.assertRaises(ValueError):
                load_conllu(file_path)
        finally:
            file_path.unlink()

    def test_rejects_invalid_token_id(self):
        invalid_line = "abc\tword\t_\tNOUN\t_\t_\t0\troot\t_\t_\n"
        file_path = self.write_conllu(invalid_line)

        try:
            with self.assertRaises(ValueError):
                load_conllu(file_path)
        finally:
            file_path.unlink()


if __name__ == "__main__":
    unittest.main()