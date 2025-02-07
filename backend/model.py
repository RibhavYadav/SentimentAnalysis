import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import pandas as pd
import time
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets.data_model import SentimentDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_json("./datasets/IMDB-Processed.json", orient="records", lines=True)

# Build vocabulary
counter = Counter()
for token in df["tokens"]:
    counter.update(token)

# Assign indices to most common words
MAX_VOCAB_SIZE = 50000
vocab = {"<PAD>": 0, "<UNK>": 1}
for word, _ in counter.most_common(MAX_VOCAB_SIZE - len(vocab)):
    vocab[word] = len(vocab)

# Split data for training and testing
train_text, test_text, train_label, test_label = train_test_split(df["input_ids"], df["sentiment"], random_state=42)

# Create datasets
train_dataset = SentimentDataset(train_text.tolist(), train_label.tolist(), vocab)
test_dataset = SentimentDataset(test_text.tolist(), test_label.tolist(), vocab)

# Define batch size for dataloaders
BATCH_SIZE = 8
train_loader = train_dataset.get_dataloader(batch_size=BATCH_SIZE)
test_loader = test_dataset.get_dataloader(batch_size=BATCH_SIZE)


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                if "embedding" in name:
                    init.uniform_(param, -0.1, 0.1)
                elif "lstm" in name:
                    init.xavier_uniform_(param)
                elif "fc" in name:
                    init.xavier_uniform_(param)
            elif "bias" in name:
                init.constant_(param, 0)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.sigmoid(self.fc(hidden[-1]))


# Initialize Model
print("Initializing Model")
model = SentimentModel(len(vocab), embedding_dim=100, hidden_dim=128)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.to(device)
criterion.to(device)

acc = 0
start_time, end_time = 0, 0
# Training Loop
print("Starting Training\n")
for epoch in range(1):
    start_time = time.time()
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(texts).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device).float()
            preds = model(texts).squeeze()
            preds = (preds > 0.5).int()  # Threshold at 0.5 for binary classification
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    end_time = time.time()
    print("----------------------------------------------------")
    print(f"Epoch: {epoch + 1}\nLoss: {total_loss / len(train_loader):.6f}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Time taken: {end_time - start_time:.2f}s")
    print("----------------------------------------------------\n")

# Save Model
if acc * 100 > 75:
    torch.save(model.state_dict(), "./models/model.pt")
    print("Model saved")
