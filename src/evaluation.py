"""Evaluation metrics for sequence labeling and POS tagging."""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.preprocessing import UNKNOWN_WORD


def flatten_sequences(sequences: list[list[str]]) -> list[str]:
    """Flatten nested sentence-level token sequences."""
    return [token for seq in sequences for token in seq]


def evaluate_predictions(
    gold_sequences: list[list[str]],
    predicted_sequences: list[list[str]],
    word_sequences: list[list[str]],
    training_vocabulary: set[str] | frozenset[str],
    labels: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Compute overall accuracy, macro-F1, per-tag metrics, and OOV accuracy."""
    if not (len(gold_sequences) == len(predicted_sequences) == len(word_sequences)):
        raise ValueError("Mismatched sentence sequence counts")

    gold_tokens = flatten_sequences(gold_sequences)
    pred_tokens = flatten_sequences(predicted_sequences)
    words = flatten_sequences(word_sequences)

    if len(gold_tokens) != len(pred_tokens) or len(words) != len(gold_tokens):
        raise ValueError("Token sequence lengths do not match")
    if not gold_tokens:
        raise ValueError("Cannot evaluate empty sequences")

    # Exclude <UNK> to strictly evaluate empirical unseen tokens
    clean_vocab = set(training_vocabulary) - {UNKNOWN_WORD}
    unknown_indices = [i for i, w in enumerate(words) if w not in clean_vocab]

    if unknown_indices:
        unk_gold = [gold_tokens[i] for i in unknown_indices]
        unk_pred = [pred_tokens[i] for i in unknown_indices]
        unknown_acc = accuracy_score(unk_gold, unk_pred)
    else:
        unknown_acc = 0.0

    target_labels = list(labels) if labels is not None else sorted(set(gold_tokens) | set(pred_tokens))
    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        gold_tokens,
        pred_tokens,
        labels=target_labels,
        zero_division=0,
    )

    return {
        "token_accuracy": accuracy_score(gold_tokens, pred_tokens),
        "macro_f1": f1_score(gold_tokens, pred_tokens, average="macro", zero_division=0),
        "per_tag_precision": dict(zip(target_labels, map(float, precisions))),
        "per_tag_recall": dict(zip(target_labels, map(float, recalls))),
        "per_tag_f1": dict(zip(target_labels, map(float, f1s))),
        "unknown_word_accuracy": unknown_acc,
        "unknown_word_count": len(unknown_indices),
        "classification_report": classification_report(
            gold_tokens,
            pred_tokens,
            labels=labels,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            gold_tokens,
            pred_tokens,
            labels=labels,
        ),
    }