import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_loader import load_conllu, extract_words_tags, validate_sentences

train = load_conllu("data/train.conllu")
dev = load_conllu("data/dev.conllu")
test = load_conllu("data/test.conllu")

validate_sentences(train)
validate_sentences(dev)
validate_sentences(test)

train_words, train_tags = extract_words_tags(train)

print("Train sentences:", len(train))
print("Dev sentences:", len(dev))
print("Test sentences:", len(test))

print("\nFirst sentence:")
print(train[0])

print("\nFirst sentence words:")
print(train_words[0])

print("\nFirst sentence tags:")
print(train_tags[0])