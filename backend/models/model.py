import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from sklearn.metrics import accuracy_score


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, threshold: float = 0.5, lstm_layers: int = 2):
        super(SentimentModel, self).__init__()
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Bidirectional LSTM Layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=0.3,
                            bidirectional=True)

        # Attention Mechanism
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.softmax = nn.Softmax(dim=1)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim * 2, 1)

        # Activation
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
                elif "fc" in name or "attn" in name:
                    init.xavier_uniform_(param)
            elif "bias" in name:
                init.constant_(param, 0)

    def forward(self, x):
        # Embedding
        x = self.embedding(x)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention scores
        attn_scores = self.attn(lstm_out)
        attn_scores_weighted = self.softmax(attn_scores)
        context = torch.sum(attn_scores_weighted * lstm_out, dim=1)

        # Final output
        output = self.dropout(context) if self.training else context
        return self.sigmoid(self.fc(output))

    def train_test(self, train_loader, test_loader, batch_size: int, inputs: int, epochs: int = 10, save: bool = False):
        # Define loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        # Move the model and the criterion to the GPU
        self.to(self.device)
        criterion.to(self.device)

        # Batches and accuracy
        total_batches = inputs // batch_size
        acc, max_acc = 0, 0
        print("Starting Training\n")
        for epoch in range(epochs):
            start_time = time.time()
            self.train()
            training_loss = 0
            batch = 0

            # Training loop
            for texts, labels in train_loader:
                batch_time = time.time() - start_time
                finish_time = (batch_time / (batch + 1)) * (total_batches - (batch + 1))
                print(
                    f"\rBatch {batch + 1} / {total_batches} | Time: {batch_time:.2f} | Expected Time: {finish_time:.2f}",
                    end="")
                texts, labels = texts.to(self.device), labels.to(self.device).float()
                optimizer.zero_grad()
                outputs = self(texts).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
                batch += 1

            # Evaluation loop
            self.eval()
            testing_loss = 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for texts, labels in test_loader:
                    batch_time = time.time() - start_time
                    finish_time = (batch_time / (batch + 1)) * (total_batches - (batch + 1))
                    print(
                        f"\rBatch {batch + 1} / {total_batches} | Time: {batch_time:.2f} | Expected Time: {finish_time:.2f}",
                        end="")
                    texts, labels = texts.to(self.device), labels.to(self.device).float()
                    preds = self(texts).squeeze()
                    testing_loss += criterion(preds, labels).item()
                    preds = (preds > self.threshold).int()
                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.tolist())
                    batch += 1

            # Accuracy calculation
            acc = accuracy_score(all_labels, all_preds)
            end_time = time.time()

            # Print statistics
            print("\n----------------------------------------------------")
            print(f"Epoch: {epoch + 1}\nTraining Loss: {training_loss / len(train_loader):.6f}")
            print(f"Testing Loss: {testing_loss / len(test_loader):.6f}\nAccuracy: {acc * 100:.2f}%")
            print(f"Time taken: {end_time - start_time:.2f}s")
            print("----------------------------------------------------\n")

            # Save Model
            if save and acc * 100 > max_acc and acc * 100 > 90:
                torch.save(self.state_dict(), "model_weights.pt")
                print("Model saved")
                max_acc = max(max_acc, acc * 100)

    def word_tokens(self, text: str, vocab) -> torch.Tensor:
        tokens = [vocab.get(word, vocab["<UNK>"]) for word in text.lower().split()]
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)

    def predict(self, text: str, vocab):
        tokens = self.word_tokens(text, vocab)
        embedding = self.embedding(tokens)
        lstm_out, _ = self.lstm(embedding)
        attn_scores = self.attn(lstm_out)
        attn_scores_weighted = self.softmax(attn_scores)
        context = torch.sum(attn_scores_weighted * lstm_out, dim=1)
        sentiment_score = self.sigmoid(self.fc(context))

        word_scores = attn_scores_weighted.squeeze().tolist()
        words = text.split()
        word_scores = {word: round(score, 3) for word, score in zip(words, word_scores)}
        word_scores = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return word_scores[:10], sentiment_score.item()
