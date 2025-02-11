import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from sklearn.metrics import accuracy_score


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                if "embedding" in name:
                    init.orthogonal_(param)
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

    def train_test(self, train_loader, test_loader, epochs: int = 10,  save: bool = False):
        # Get the GPU for training
        device = torch.device("cuda")

        # Define loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        # Move the model and the criterion to the GPU
        self.to(device)
        criterion.to(device)

        acc, max_acc = 0, 0
        print("Starting Training\n")
        for epoch in range(epochs):
            start_time = time.time()
            self.train()
            total_loss = 0

            # Training loop
            for texts, labels in train_loader:
                texts, labels = texts.to(device), labels.to(device).float()
                optimizer.zero_grad()
                outputs = self(texts).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Evaluation loop
            self.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for texts, labels in test_loader:
                    texts, labels = texts.to(device), labels.to(device).float()
                    preds = self(texts).squeeze()
                    preds = (preds > 0.5).int()
                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.tolist())

            # Accuracy calculation
            acc = accuracy_score(all_labels, all_preds)
            end_time = time.time()

            # Print statistics
            print("----------------------------------------------------")
            print(f"Epoch: {epoch + 1}\nLoss: {total_loss / len(train_loader):.6f}")
            print(f"Accuracy: {acc * 100:.2f}%")
            print(f"Time taken: {end_time - start_time:.2f}s")
            print("----------------------------------------------------\n")

            # Save Model
            if save and acc * 100 > max_acc and acc * 100 > 90:
                torch.save(self.state_dict(), "model_weights.pt")
                print("Model saved")
                max_acc = max(max_acc, acc * 100)
