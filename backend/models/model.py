import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from sklearn.metrics import accuracy_score


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, threshold: float = 0.5, lstm_layers: int = 2):
        super(SentimentModel, self).__init__()
        # Neural network layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # Dropout
        self.dropout = nn.Dropout(0.3)
        self.threshold = threshold

        # Use CUDA device if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Initialize layer weights
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

    def train_test(self, train_loader, test_loader, epochs: int = 10, save: bool = False):
        # Define loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        # Move the model and the criterion to the GPU
        self.to(self.device)
        criterion.to(self.device)

        acc, max_acc = 0, 0
        print("Starting Training\n")
        for epoch in range(epochs):
            start_time = time.time()
            self.train()
            training_loss = 0

            # Training loop
            for texts, labels in train_loader:
                texts, labels = texts.to(self.device), labels.to(self.device).float()
                optimizer.zero_grad()
                outputs = self(texts).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()

            # Evaluation loop
            self.eval()
            testing_loss = 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for texts, labels in test_loader:
                    texts, labels = texts.to(self.device), labels.to(self.device).float()
                    preds = self(texts).squeeze()
                    testing_loss += criterion(preds, labels).item()
                    preds = (preds > self.threshold).int()
                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.tolist())

            # Accuracy calculation
            acc = accuracy_score(all_labels, all_preds)
            end_time = time.time()

            # Print statistics
            print("----------------------------------------------------")
            print(f"Epoch: {epoch + 1}\nTraining Loss: {training_loss / len(train_loader):.6f}")
            print(f"Testing Loss: {testing_loss / len(test_loader):.6f}\nAccuracy: {acc * 100:.2f}%")
            print(f"Time taken: {end_time - start_time:.2f}s")
            print("----------------------------------------------------\n")

            # Save Model
            if save and acc * 100 > max_acc and acc * 100 > 90:
                torch.save(self.state_dict(), "model_weights.pt")
                print("Model saved")
                max_acc = max(max_acc, acc * 100)

    def get_tokens(self, text: str, vocab) -> torch.Tensor:
        tokens = [vocab.get(word, vocab["<UNK>"]) for word in text.lower().split()]
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)

    def predict(self, text: str, vocab):
        tokens = self.get_tokens(text, vocab)
        with torch.no_grad():
            prediction = self(tokens).squeeze().item()
        sentiment = "Positive" if prediction > self.threshold else "Negative"
        return sentiment, prediction
