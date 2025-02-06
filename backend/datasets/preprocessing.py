import torch
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
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

# Assign indices to most common word
vocab = {"<PAD>": 0, "<UNK>": 1}
for word, count in counter.most_common():
    vocab[word] = len(vocab)


def text_pipline(tokens):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]


# Indexed tokens for the reviews
df["input_ids"] = df["tokens"].apply(text_pipline)

# Save processed data
df.to_csv("IMDB-Processed.csv", index=False)
