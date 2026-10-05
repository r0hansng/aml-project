def load_conllu(path):
    sentences = []
    sentence = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")

            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue

            if line.startswith("#"):
                continue

            fields = line.split("\t")

            if len(fields) != 10:
                raise ValueError(f"Invalid CoNLL-U line: {line}")

            token_id = fields[0]
            form = fields[1]
            upos = fields[3]

            if "-" in token_id or "." in token_id:
                continue

            sentence.append((form, upos))

    if sentence:
        sentences.append(sentence)

    return sentences


def extract_words_tags(sentences):
    words = []
    tags = []

    for sentence in sentences:
        sentence_words = []
        sentence_tags = []

        for word, tag in sentence:
            sentence_words.append(word)
            sentence_tags.append(tag)

        words.append(sentence_words)
        tags.append(sentence_tags)

    return words, tags


def get_tokens(sentences):
    return [token for sentence in sentences for token in sentence]


def validate_sentences(sentences):
    if not sentences:
        raise ValueError("No sentences were loaded")

    for sentence_index, sentence in enumerate(sentences):
        if not sentence:
            raise ValueError(
                f"Empty sentence at index {sentence_index}"
            )

        for token_index, token in enumerate(sentence):
            if len(token) != 2:
                raise ValueError(
                    f"Invalid token at sentence {sentence_index}, "
                    f"token {token_index}"
                )

            word, tag = token

            if not word:
                raise ValueError(
                    f"Empty word at sentence {sentence_index}, "
                    f"token {token_index}"
                )

            if not tag:
                raise ValueError(
                    f"Empty POS tag at sentence {sentence_index}, "
                    f"token {token_index}"
                )