import pandas as pd

# Load dataset
df = pd.read_csv('amazon_reviews.csv', names=["sentiment", "review_title", "text"])

# Remove unnecessary column: review_title
df.drop("review_title", axis="columns", inplace=True)

# Map sentiment values: Negative: 1 -> 0, Positive: 2 -> 1
df["sentiment"] = df["sentiment"].map({1: 0, 2: 1})

# Rearrange columnds
df = df[["text", "sentiment"]]

# Save the processed and reduced dataset
df.to_csv("amazon_reviews_processed.csv", index=False)
