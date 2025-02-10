import pandas as pd
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize

# Load processed datasets
IMDB = pd.read_csv('./IMDB/IMDB-Dataset_processed.csv')
tweets = pd.read_csv('./Tweets/tweet_data_processed.csv')

# Combine the datasets
dataset = pd.concat([IMDB, tweets], ignore_index=True)

# Tokenize the text
dataset["tokens"] = dataset["text"].apply(word_tokenize)

# Build vocabulary for each token
tokens = Counter()
for token in dataset["tokens"]:
    tokens.update(token)

MAX_VOCAB_SIZE = 50000
vocab = {"<PAD>": 0, "<UNK>": 1}
for word, count in tokens.most_common(MAX_VOCAB_SIZE - len(vocab)):
    vocab[word] = len(vocab)

# Convert the tokens in the dataframe into their unique ID from the vocabulary
dataset["token_ids"] = dataset["tokens"].apply(lambda tokens_list: [vocab.get(token, vocab["<UNK>"]) for token in tokens_list])

# Drop redundant columns: "text", "tokens"
dataset.drop("text", axis="columns", inplace=True)
dataset.drop("tokens", axis="columns", inplace=True)

# Save vocabulary and processed data
np.save("vocab.npy", vocab)
dataset.to_json("training_data.json", orient="records", lines=True)
