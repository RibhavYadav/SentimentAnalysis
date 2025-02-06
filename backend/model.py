import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
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

# Assign indices to most common word
vocab = {"<PAD>": 0, "<UNK>": 1}
for word, count in counter.most_common():
    vocab[word] = len(vocab)

# Split data for training and testing
train_text, test_text, train_label, text_label = train_test_split(df["input_ids"], df["sentiment"])

# Create dataset
train_dataset = SentimentDataset(train_text.tolist(), train_label.tolist(), vocab)
test_dataset = SentimentDataset(test_text.tolist(), text_label.tolist(), vocab)

# Get data loaders
train_loader = train_dataset.get_dataloader()
test_loader = test_dataset.get_dataloader()


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.sigmoid(self.fc(hidden[-1]))


# Create and train
print("Initialising Model")
model = SentimentModel(len(vocab), 100, 128)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.to(device)
criterion.to(device)

print("Starting Evaluation\n")
for epoch in range(10):
    model.train()
    loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss += loss.item()
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            preds = model(texts).squeeze()
            preds = (preds > 0.5).float()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    acc = accuracy_score(all_labels, all_preds)
    print("----------------------------------------------------")
    print(f"Epoch: {epoch + 1}, Loss: {loss / len(train_loader):.10f}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("----------------------------------------------------\n")

torch.save(model.state_dict(), "./models/model.pt")
