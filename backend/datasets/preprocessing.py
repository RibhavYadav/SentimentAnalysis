import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize

# Load data and tokenize text
df = pd.read_csv('./IMDB-Dataset.csv')
df["tokens"] = df["review"].apply(word_tokenize)

# Assign values to the sentiment
df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1})

# Build vocabulary
counter = Counter()
for token in df["tokens"]:
    counter.update(token)

# Assign indices to most common words
MAX_VOCAB_SIZE = 50000
vocab = {"<PAD>": 0, "<UNK>": 1}
for word, _ in counter.most_common(MAX_VOCAB_SIZE - len(vocab)):
    vocab[word] = len(vocab)

# Convert tokens to input IDs using vocab
df["input_ids"] = df["tokens"].apply(lambda tokens: [vocab.get(token, vocab["<UNK>"]) for token in tokens])


def text_pipline(tokens):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]


# Indexed tokens for the reviews
df["input_ids"] = df["tokens"].apply(text_pipline)

# Save processed data
df.to_json("IMDB-Processed.json", orient="records", lines=True)
