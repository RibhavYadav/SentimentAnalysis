import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, threshold: float = 0.8, lstm_layers: int = 2):
        super(SentimentModel, self).__init__()
        # Neural network layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=0.3,
                            bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 1)
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
                elif "lstm" in name or "fc" in name or "attention" in name:
                    init.xavier_uniform_(param)
            elif "bias" in name:
                init.constant_(param, 0)

    def attention_layer(self, lstm_output):
        attn_weights = torch.tanh(self.attention(lstm_output)).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(-1)
        attention = torch.sum(lstm_output * attn_weights, dim=1)
        return attention, attn_weights.squeeze(-1)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        attended_output, _ = self.attention_layer(lstm_out)  # Ignore attention weights during training
        output = self.dropout(self.fc(attended_output)) if self.training else self.fc(attended_output)
        return self.sigmoid(output)

    def train_test(self, train_loader, test_loader, epochs: int = 10, save: bool = False):

        # Define loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        # Move the model and criterion to the GPU
        self.to(self.device)
        criterion.to(self.device)

        total_batches = len(train_loader)
        max_acc = 0
        print("Starting Training\n")

        for epoch in range(epochs):
            start_time = time.time()
            self.train()
            training_loss = 0

            # Training loop
            for batch, (texts, labels) in enumerate(train_loader, start=1):
                texts, labels = texts.to(self.device), labels.to(self.device).float()
                optimizer.zero_grad()
                outputs = self(texts).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()

                elapsed_time = time.time() - start_time
                remaining_batches = total_batches - batch
                expected_time = (elapsed_time / batch) * remaining_batches if batch > 0 else 0
                print(
                    f"\rBatch {batch} / {total_batches:<10} | Elapsed Time: {elapsed_time:.2f}s | Expected Time: {expected_time:.2f}s",
                    end="")

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
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())

            # Compute metrics
            acc = accuracy_score(all_labels, all_preds) * 100
            precision = precision_score(all_labels, all_preds, zero_division=0) * 100
            recall = recall_score(all_labels, all_preds, zero_division=0) * 100
            f1 = f1_score(all_labels, all_preds, zero_division=0) * 100
            epoch_time = time.time() - start_time

            # Print statistics
            print("\n" + "-" * 100)
            print(f"Epoch: {epoch + 1}\nTraining Loss: {training_loss / len(train_loader):.6f}")
            print(f"Testing Loss: {testing_loss / len(test_loader):.6f}")
            print(f"Accuracy: {acc:.2f}% | Precision: {precision:.2f}% | Recall: {recall:.2f}% | F1-score: {f1:.2f}%")
            print(f"Time taken: {epoch_time:.2f}s")
            print("-" * 100 + "\n")

            # Save Model
            if save and acc * 100 > max_acc and acc * 100 > 90:
                torch.save(self.state_dict(), "model_weights.pt")
                print("Model saved")
                max_acc = max(max_acc, acc * 100)

    def predict(self, text: str, vocab):
        tokens = self.get_tokens(text, vocab)
        with torch.no_grad():
            lstm_out, _ = self.lstm(self.embedding(tokens))
            attended_output, attn_weights = self.attention_layer(lstm_out)
            prediction = self.sigmoid(self.fc(attended_output)).squeeze()
        sentiment = "Positive" if prediction.item() > self.threshold else "Negative"
        return sentiment, prediction.item(), attn_weights.cpu().numpy()

    def get_tokens(self, text: str, vocab) -> torch.Tensor:
        tokens = [vocab.get(word, vocab["<UNK>"]) for word in text.lower().split()]
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
