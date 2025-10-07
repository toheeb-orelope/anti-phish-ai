# CNN with PyTorch Lightning
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# -----------------------------
# Dataset
# -----------------------------
class URLDataset(Dataset):
    def __init__(self, urls, labels, max_len=200):
        self.max_len = max_len
        self.urls = [self.encode_url(u) for u in urls]
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def encode_url(self, url):
        url = str(url)[: self.max_len].ljust(self.max_len)
        return torch.tensor([ord(c) / 128 for c in url], dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.urls[idx], self.labels[idx]


# -----------------------------
# Lightning Model
# -----------------------------
class LightningCNN(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 50, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.lr = lr
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Buffers to store validation outputs
        self.validation_step_outputs = []

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, len]
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x).squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        # Log both loss and current LR
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        preds = torch.sigmoid(y_hat.detach())

        # Store batch results
        self.validation_step_outputs.append(
            {
                "val_loss": loss.detach(),
                "preds": preds.cpu(),
                "targets": y.cpu(),
            }
        )

    def on_validation_epoch_end(self):
        # Aggregate all outputs
        avg_loss = torch.stack(
            [x["val_loss"] for x in self.validation_step_outputs]
        ).mean()
        preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
        targets = torch.cat([x["targets"] for x in self.validation_step_outputs])

        try:
            auc = roc_auc_score(targets.numpy(), preds.numpy())
        except ValueError:
            auc = 0.5

        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_auc", auc, prog_bar=True)
        print(
            f"\nEpoch {self.current_epoch}: val_loss={avg_loss:.4f}, val_auc={auc:.4f}"
        )

        # Clear for next epoch
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# -----------------------------
# Main Training Entry Point
# -----------------------------
if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Load data
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    # Datasets and Dataloaders
    train_ds = URLDataset(train["url"], train["label"])
    test_ds = URLDataset(test["url"], test["label"])

    train_loader = DataLoader(
        train_ds,
        batch_size=256,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        test_ds, batch_size=256, num_workers=2, pin_memory=True, persistent_workers=True
    )

    # Initialize model
    model = LightningCNN(lr=1e-3)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",  # Automatically uses GPU if available
        devices=1,
        precision="16-mixed",  # ✅ Safe AMP precision
        log_every_n_steps=10,
        num_sanity_val_steps=0,  # Skip initial validation sanity checks
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="models",
                filename="cnn_best",
                monitor="val_auc",
                mode="max",
                save_top_k=1,
            ),
            pl.callbacks.EarlyStopping(monitor="val_auc", patience=3, mode="max"),
        ],
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Save final checkpoint
    trainer.save_checkpoint("models/cnn_lightning.ckpt")
