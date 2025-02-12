import pandas as pd

# Download dataset from here: https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/

# Load dataset
df = pd.read_csv('./train.csv', names=["sentiment", "review_title", "review"])

# Remove unnecessary column: review_title
df.drop("review_title", axis="columns", inplace=True)

# Map sentiment values: Negative: 1 -> 0, Positive: 2 -> 1
df["sentiment"] = df["sentiment"].map({1: 0, 2: 1})

# Save processed data
df.to_csv("amazon_reviews_processed.csv", index=False)
