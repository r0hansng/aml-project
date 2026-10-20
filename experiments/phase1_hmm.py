"""Train, tune, and evaluate the Phase 1 HMM POS tagger on UD English-EWT."""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_loader import extract_words_tags, load_conllu
from src.dataset_registry import DATASET_SPLITS, validate_dataset_files
from src.evaluation import evaluate_predictions
from src.hmm_model import HMMPOSTagger
from src.preprocessing import extract_original_vocabulary


def main():
    validate_dataset_files()

    train_sentences = load_conllu(DATASET_SPLITS["train"])
    dev_sentences = load_conllu(DATASET_SPLITS["dev"])
    test_sentences = load_conllu(DATASET_SPLITS["test"])

    train_words, train_tags = extract_words_tags(train_sentences)
    dev_words, dev_tags = extract_words_tags(dev_sentences)
    test_words, test_tags = extract_words_tags(test_sentences)

    original_training_vocab = extract_original_vocabulary(train_sentences)

    # 1. Dev set hyperparameter tuning over add-alpha smoothing
    candidate_alphas = (0.01, 0.1, 1.0)
    best_model = None
    best_dev_macro_f1 = float("-inf")
    best_alpha = None

    print("=== Hyperparameter Tuning on Development Set ===")
    for alpha in candidate_alphas:
        model = HMMPOSTagger(smoothing=alpha, minimum_word_frequency=1).fit(train_sentences)
        dev_preds = model.predict_batch(dev_words)

        dev_metrics = evaluate_predictions(
            gold_sequences=dev_tags,
            predicted_sequences=dev_preds,
            word_sequences=dev_words,
            training_vocabulary=original_training_vocab,
        )

        print(
            f"  alpha={alpha:<5} -> "
            f"Token Acc: {dev_metrics['token_accuracy']:.4f} | "
            f"Macro F1: {dev_metrics['macro_f1']:.4f} | "
            f"Unknown Acc: {dev_metrics['unknown_word_accuracy']:.4f}"
        )

        if dev_metrics["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = dev_metrics["macro_f1"]
            best_model = model
            best_alpha = alpha

    print(f"\nSelected optimal smoothing from Dev: alpha = {best_alpha}")

    # 2. Final evaluation on unseen test set
    print("\n=== Final Evaluation on Test Set ===")
    test_preds = best_model.predict_batch(test_words)

    test_metrics = evaluate_predictions(
        gold_sequences=test_tags,
        predicted_sequences=test_preds,
        word_sequences=test_words,
        training_vocabulary=original_training_vocab,
    )

    print(f"Model: First-Order HMM (Smoothing alpha = {best_alpha})")
    print(f"Test Token Accuracy:          {test_metrics['token_accuracy']:.4f}")
    print(f"Test Macro-averaged F1:       {test_metrics['macro_f1']:.4f}")
    print(f"Test Unknown-Word Accuracy:   {test_metrics['unknown_word_accuracy']:.4f}")
    print(f"Test Unknown Tokens:          {test_metrics['unknown_word_count']:,}")

    print("\nClassification Report:\n")
    print(test_metrics["classification_report"])

    # Error analysis: Top confusion pairs (Gold -> Pred)
    error_counts = Counter(
        f"{gold_tag} -> {pred_tag}"
        for g_sent, p_sent in zip(test_tags, test_preds)
        for gold_tag, pred_tag in zip(g_sent, p_sent)
        if gold_tag != pred_tag
    )

    print("--- Top Error Confusion Patterns (Gold -> Predicted) ---")
    for pair, count in error_counts.most_common(5):
        print(f"  {pair:<18}: {count} errors")


if __name__ == "__main__":
    main()