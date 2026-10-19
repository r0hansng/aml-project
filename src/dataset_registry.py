from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"

DATASET_SPLITS = {
    "train": DATA_DIRECTORY / "train.conllu",
    "dev": DATA_DIRECTORY / "dev.conllu",
    "test": DATA_DIRECTORY / "test.conllu",
}

def validate_dataset_files():
    """Raise FileNotFoundError if an official split is missing."""
    missing_files = [
        split_path
        for split_path in DATASET_SPLITS.values()
        if not split_path.is_file()
    ]

    if missing_files:
        missing_names = ", ".join(
            str(file_path) for file_path in missing_files
        )
        raise FileNotFoundError(
            f"Missing official dataset files: {missing_names}"
        )