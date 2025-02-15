import swifter
import pandas as pd
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize

# Process Amazon dataset in chunks
amazon_iter = pd.read_csv('./Amazon_Reviews/amazon_reviews_processed.csv', chunksize=100000)
amazon = pd.concat(amazon_iter, ignore_index=True)

# Load IMDB dataset
imdb = pd.read_csv('./IMDB/IMDB-Dataset_processed.csv')

# Combine both datasets
dataset = pd.concat([imdb, amazon], axis=0, ignore_index=True)

# Tokenize text using faster approach
dataset["tokens"] = dataset["text"].swifter.apply(lambda x: word_tokenize(x.lower()))
print("Processed Tokens")

# Drop original text column to free memory
dataset.drop("text", axis=1, inplace=True)

# Build vocabulary
tokens = Counter(token for sublist in dataset["tokens"] for token in sublist)

MAX_VOCAB_SIZE = 50000
vocab = {"<PAD>": 0, "<UNK>": 1}
for word, _ in tokens.most_common(MAX_VOCAB_SIZE - len(vocab)):
    vocab[word] = len(vocab)
print("Vocab built")

# Convert words to token IDs (use uint16 to reduce memory usage)
dataset["token_ids"] = dataset["tokens"].swifter.apply(
    lambda tokens_list: np.array([vocab.get(token, vocab["<UNK>"]) for token in tokens_list], dtype=np.uint16)
)
print("Processed Token IDs")

# Drop tokens column to save memory
dataset.drop("tokens", axis=1, inplace=True)

# Save vocabulary and processed data
np.save("vocab.npy", vocab)
dataset.to_json("training_data.json", orient="records", lines=True)
