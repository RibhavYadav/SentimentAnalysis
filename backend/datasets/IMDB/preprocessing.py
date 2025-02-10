import pandas as pd

# Load dataset
df = pd.read_csv('IMDB-Dataset.csv')

# Map sentiment Negative -> 0, Positive -> 1
df["sentiment"] = df["sentiment"].map({'positive': 1, 'negative': 0})

# Save processed file
df.to_csv('IMDB-Dataset_processed.csv', index=False)
