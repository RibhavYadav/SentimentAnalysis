import pandas as pd

# Load dataset
df = pd.read_csv('amazon_reviews.csv', names=["sentiment", "review_title", "text"])

# Remove unnecessary column: review_title
df.drop("review_title", axis="columns", inplace=True)

# Map sentiment values: Negative: 1 -> 0, Positive: 2 -> 1
df["sentiment"] = df["sentiment"].map({1: 0, 2: 1})

# Separate dataset into sentiment 0 and sentiment 1
df_sentiment_0 = df[df["sentiment"] == 0]
df_sentiment_1 = df[df["sentiment"] == 1]

# Randomly sample 500,000 rows from each class
df_sentiment_0_sampled = df_sentiment_0.sample(n=500000, random_state=42)
df_sentiment_1_sampled = df_sentiment_1.sample(n=500000, random_state=42)

# Combine the sampled data
df_sampled = pd.concat([df_sentiment_0_sampled, df_sentiment_1_sampled])

# Shuffle the combined dataset
df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the processed and reduced dataset
df_sampled.to_csv("amazon_reviews_processed.csv", index=False)
