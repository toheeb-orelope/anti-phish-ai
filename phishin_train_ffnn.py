# Feedforward Neural Network (FFNN) for Phishing URL Detection
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import os
from vis_metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall


# -----------------------------
# Dataset (reuse same structure as CNN)
# -----------------------------
class URLDataset(Dataset):
    def __init__(self, urls, labels, max_len=200):
        self.max_len = max_len
        self.urls = [self.encode_url(u) for u in urls]
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def encode_url(self, url):
        url = str(url)[: self.max_len].ljust(self.max_len)
        encoded = [ord(c) / 128 for c in url]
        return torch.tensor(encoded, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.urls[idx], self.labels[idx]


# -----------------------------
# Feedforward Neural Network
# -----------------------------
class LightningFFNN(pl.LightningModule):
    def __init__(self, input_size=200, hidden_size=256, lr=1e-3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.lr = lr
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x).squeeze()

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
        try:
            auc = roc_auc_score(y.cpu(), y_hat.detach().cpu())
        except ValueError:
            auc = 0.0
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_auc", auc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# -----------------------------
# Training
# -----------------------------
def train_ffnn():
    print("[*] Loading data...")
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

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

    model = LightningFFNN()

    print("[*] Training FFNN...")
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",  # automatically use GPU if available
        devices=1,
        precision="16-mixed",
        log_every_n_steps=10,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="models",
                filename="ffnn_best",
                monitor="val_auc",
                mode="max",
                save_top_k=1,
            ),
            pl.callbacks.EarlyStopping(monitor="val_auc", patience=3, mode="max"),
        ],
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint("models/ffnn_lightning.ckpt")

    print("[+] FFNN training complete. Model saved to models/ffnn_lightning.ckpt")

    # -----------------------------
    # Evaluation and Metrics
    # -----------------------------
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for x, y in val_loader:
            probs = model(x)
            preds = (probs > 0.5).int()
            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())
            y_prob.extend(probs.numpy())

    os.makedirs("plots", exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, save_path="plots/ffnn_confusion_matrix.png")
    plot_roc_curve(
        y_true, y_prob, model_name="FFNN", save_path="plots/ffnn_roc_curve.png"
    )
    plot_precision_recall(
        y_true, y_prob, model_name="FFNN", save_path="plots/ffnn_pr_curve.png"
    )

    print("[+] Plots saved in 'plots/' folder.")


if __name__ == "__main__":
    train_ffnn()
