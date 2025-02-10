import torch
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets.data_model import SentimentDataset

device = torch.device("cuda")

# Load data
df = pd.read_json("../datasets/IMDB/IMDB-Processed.json", orient="records", lines=True)
vocab = np.load("../datasets/IMDB/vocab.npy", allow_pickle=True).item()

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
        self.dropout = nn.Dropout(0.3)
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
        output = self.dropout(hidden[-1]) if self.training else hidden[-1]
        return self.sigmoid(self.fc(output))


# Initialize Model
print("Initializing Model")
model = SentimentModel(len(vocab), embedding_dim=100, hidden_dim=128)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.to(device)
criterion.to(device)

acc, max_acc = 0, 0
start_time, end_time = 0, 0
# Training Loop
print("Starting Training\n")
for epoch in range(10):
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
            preds = (preds > 0.5).int()
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
    if acc > max_acc and acc * 100 > 90:
        torch.save(model.state_dict(), "model_weights.pt")
        print("Model saved")
        max_acc = max(max_acc, acc * 100)
