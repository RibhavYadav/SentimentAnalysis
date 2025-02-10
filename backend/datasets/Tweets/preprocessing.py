import pandas as pd

# Load data and assign column names
column_names = ["sentiment", "id", "date", "flag", "user", "text"]
df = pd.read_csv("tweet_data.csv", names=column_names, header=None, encoding="latin1")

# Drop ID, date, user and flag as it isn't required for sentiment analysis
df.drop("id", axis="columns", inplace=True)
df.drop("date", axis="columns", inplace=True)
df.drop("flag", axis="columns", inplace=True)
df.drop("user", axis="columns", inplace=True)

# Map sentiment Negative: 0 -> 0, Positive: 4 -> 1
df["sentiment"] = df["sentiment"].map({0: 0, 4: 1})

# Rearrange columns
df = df[["text", "sentiment"]]

# Save
df.to_csv("tweet_data_processed.csv", index=False)
