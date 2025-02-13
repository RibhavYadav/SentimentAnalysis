import warnings
import numpy as np
import pandas as pd
from model import SentimentModel
from sklearn.model_selection import train_test_split
from datasets.data_model import SentimentDataset

# Suppress specific UserWarnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*To copy construct from a tensor.*")

# Load data
df = pd.read_json("../datasets/training_data.json", orient="records", lines=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
vocab = np.load("../datasets/vocab.npy", allow_pickle=True).item()

# Split data for training and testing
train_text, test_text, train_label, test_label = train_test_split(df["token_ids"], df["sentiment"], random_state=42)

# Create datasets
train_dataset = SentimentDataset(train_text.tolist(), train_label.tolist(), vocab)
test_dataset = SentimentDataset(test_text.tolist(), test_label.tolist(), vocab)

# Define batch size for dataloaders
BATCH_SIZE = 16
train_loader = train_dataset.get_dataloader(batch_size=BATCH_SIZE)
test_loader = test_dataset.get_dataloader(batch_size=BATCH_SIZE)

# Initialize Model
print("Initializing Model")
model = SentimentModel(len(vocab), embedding_dim=100, hidden_dim=128)

# Train the model
model.train_test(train_loader, test_loader, save=True)
