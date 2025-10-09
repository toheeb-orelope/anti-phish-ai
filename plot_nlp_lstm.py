# evaluate_lstm.py
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from phishin_nlp_lstm import LightningLSTM, URLDataset
from vis_metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall


def evaluate_lstm():
    print("[*] Loading data...")
    test = pd.read_csv("data/processed/test.csv")
    test_ds = URLDataset(test["url"], test["label"])
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=0, pin_memory=True)

    print("[*] Loading trained LSTM model...")
    model = LightningLSTM.load_from_checkpoint("models/lstm_lightning.ckpt")
    model.eval()

    print("[*] Evaluating...")
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            probs = torch.sigmoid(logits)  # convert logits to probabilities
            preds = (probs > 0.5).int()
            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())
            y_prob.extend(probs.numpy())

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n=== LSTM Evaluation ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Save plots
    os.makedirs("plots", exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, save_path="plots/lstm_confusion_matrix.png")
    plot_roc_curve(
        y_true, y_prob, model_name="LSTM", save_path="plots/lstm_roc_curve.png"
    )
    plot_precision_recall(
        y_true, y_prob, model_name="LSTM", save_path="plots/lstm_pr_curve.png"
    )

    print("\n[+] Plots saved to 'plots/' folder.")


if __name__ == "__main__":
    evaluate_lstm()
