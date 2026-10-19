# Project 16: Part-of-Speech Tagging

## Title
**Part-of-Speech Tagging via Hidden Markov Models and Bidirectional LSTMs**

## Project Overview and Abstract

Assign grammatical tags to words in sentences while comparing a generative sequence model decoded by the Viterbi algorithm with a neural sequence tagger that uses context from both directions.

### Hybrid Logic: Markov Emission Assumptions to Contextual Sequence Representation

- **AML baseline:** Estimate Hidden Markov Model (HMM) transition and emission probabilities with smoothing, then decode sequences using the Viterbi algorithm.
- **Identify the gap:** An HMM assumes a limited Markov context and can struggle with unknown words and long-range syntactic dependencies.
- **Deep learning model:** Train a Bidirectional Long Short-Term Memory (BiLSTM) tagger with word and character or subword embeddings. Mask padding tokens in the loss function.
- **Fair comparison:** Use the standard train, development, and test split. Report token accuracy, macro F1-score, unknown-word accuracy, and a small error analysis by tag.

## Suggested Pre-reads

- Rabiner (1989), *A Tutorial on Hidden Markov Models*.
- Huang et al. (2015), *Bidirectional LSTM-CRF Models for Sequence Tagging*.

## Dataset and Technology Stack

### Recommended Starter Dataset

**Universal Dependencies English EWT Treebank**, using its published train, development, and test split.

- **Dataset:** Universal Dependencies English EWT
- **Dataset link:** Official treebank repository

### Suggested Stack

- Python
- `conllu`
- NumPy
- PyTorch
- scikit-learn
- Matplotlib

## Feature Engineering and Setup Tips

- Preserve sentence boundaries.
- Build the vocabulary from the training split only and handle unseen words using an `UNK` token.
- Add start and end states to the Viterbi algorithm.
- Pad batches and mask padding tokens correctly.
- Preserve the official part-of-speech tag mapping.

## Expected Deliverable

A sequence-tagging benchmark comparing HMM and BiLSTM results, including:

- Model performance results.
- Unknown-word analysis.
- Decoded sentence examples.
- Error analysis by part-of-speech tag.