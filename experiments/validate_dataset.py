"""Validate that the official dataset splits are available and loadable."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_loader import load_conllu
from src.dataset_registry import DATASET_SPLITS, validate_dataset_files


validate_dataset_files()

train_sentences = load_conllu(DATASET_SPLITS["train"])
dev_sentences = load_conllu(DATASET_SPLITS["dev"])
test_sentences = load_conllu(DATASET_SPLITS["test"])