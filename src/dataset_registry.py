"""Dataset path registry and verification for Universal Dependencies EWT."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"

DATASET_SPLITS = {
    "train": DATA_DIRECTORY / "train.conllu",
    "dev": DATA_DIRECTORY / "dev.conllu",
    "test": DATA_DIRECTORY / "test.conllu",
}


def validate_dataset_files():
    """Ensure all official dataset split files exist on disk."""
    missing = [path for path in DATASET_SPLITS.values() if not path.is_file()]
    if missing:
        missing_names = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing dataset files: {missing_names}")