"""Run a basic validation check for the project's CoNLL-U datasets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_loader import (
	extract_words_tags,
	load_conllu,
	validate_sentences,
)

train_sentences = load_conllu("data/train.conllu")
dev_sentences = load_conllu("data/dev.conllu")
test_sentences = load_conllu("data/test.conllu")

validate_sentences(train_sentences)
validate_sentences(dev_sentences)
validate_sentences(test_sentences)

train_words, train_tags = extract_words_tags(train_sentences)

print("Train sentences:", len(train_sentences))
print("Dev sentences:", len(dev_sentences))
print("Test sentences:", len(test_sentences))

print("\nFirst sentence:")
print(train_sentences[0])

print("\nFirst sentence words:")
print(train_words[0])

print("\nFirst sentence tags:")
print(train_tags[0])