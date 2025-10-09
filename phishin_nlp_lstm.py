# Natural Language Processing (NLP) LSTM (Long Short-Term Memory)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score


# -----------------------------
# Dataset
# -----------------------------
class URLDataset(Dataset):
    def __init__(self, urls, labels, max_len=200):
        self.max_len = max_len
        self.urls = [self.encode_url(u) for u in urls]
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def encode_url(self, url):
        # Clip or replace out-of-range characters
        url = str(url)[: self.max_len].ljust(self.max_len)
        ascii_indices = [min(ord(c), 127) for c in url]  # ensure within [0, 127]
        return torch.tensor(ascii_indices, dtype=torch.long)

    def __getitem__(self, idx):
        return self.urls[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


# -----------------------------
# Lightning LSTM Model
# -----------------------------


class LightningLSTM(pl.LightningModule):
    def __init__(
        self, vocab_size=128, embed_dim=32, hidden_dim=128, num_layers=2, lr=1e-3
    ):
        super().__init__()
        self.save_hyperparameters()

        # 🔹 Character embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 🔹 LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
        )

        # 🔹 Fully connected output
        self.fc = nn.Linear(hidden_dim, 1)

        # 🔹 Use BCEWithLogitsLoss (includes sigmoid internally)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, x):
        # Input shape: [batch, seq_len] (already numeric)
        x = x.long()  # required for embedding
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.squeeze()  # ⚠️ No sigmoid here

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        # 🔹 Apply sigmoid for metrics only
        y_prob = torch.sigmoid(y_hat).detach().cpu()
        auc = roc_auc_score(y.cpu(), y_prob)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_auc", auc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# -----------------------------
# Training
# -----------------------------
def train_lstm():
    print("[*] Loading dataset...")
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    train_ds = URLDataset(train["url"], train["label"])
    test_ds = URLDataset(test["url"], test["label"])

    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True
    )
    val_loader = DataLoader(
        test_ds, batch_size=256, num_workers=4, persistent_workers=True
    )

    print(
        f"✅ Training on: {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    model = LightningLSTM()

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",  # force GPU
        devices=1,
        precision="16-mixed",
        log_every_n_steps=10,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="models",
                filename="lstm_best",
                monitor="val_auc",
                mode="max",
                save_top_k=1,
            ),
            pl.callbacks.EarlyStopping(monitor="val_auc", patience=3, mode="max"),
        ],
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint("models/lstm_lightning.ckpt")
    print("\n[+] LSTM training complete. Model saved to models/lstm_lightning.ckpt")


if __name__ == "__main__":
    train_lstm()
