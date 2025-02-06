import torch
from torch.utils.data import Dataset, DataLoader


# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, text, labels, vocab):
        self.text = text
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.vocab = vocab

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text[idx], self.labels[idx]

    def collate_fn(self, batch):
        text, labels = zip(*batch)
        max_len = max(len(t) for t in text)
        padded_texts = [t + [self.vocab["<PAD>"]] * (max_len - len(t)) for t in text]
        return torch.tensor(padded_texts, dtype=torch.long), torch.tensor(labels, dtype=torch.float32)

    def get_dataloader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size, shuffle, collate_fn=self.collate_fn)
