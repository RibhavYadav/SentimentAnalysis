import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class SentimentDataset(Dataset):
    def __init__(self, text, labels, vocab):
        self.text = text
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.vocab = vocab

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = torch.tensor(self.text[idx], dtype=torch.long)
        label = self.labels[idx]
        return text, label

    def collate_fn(self, batch):
        text, labels = zip(*batch)
        text = [torch.tensor(t, dtype=torch.long).clone().detach() for t in text]
        padded_texts = pad_sequence(text, batch_first=True, padding_value=self.vocab["<PAD>"])
        return padded_texts, torch.tensor(labels, dtype=torch.float32)

    def get_dataloader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size, shuffle, collate_fn=self.collate_fn, pin_memory=True)
