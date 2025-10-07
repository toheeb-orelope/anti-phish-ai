import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from phishin_train_cnn import LightningCNN, URLDataset
from vis_metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall


def evaluate_model():
    # -------------------------
    # Load Test Dataset
    # -------------------------
    test = pd.read_csv("data/processed/test.csv")
    test_ds = URLDataset(test["url"], test["label"])
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=0, pin_memory=True)
    # 🔹 set num_workers=0 for safety on Windows

    # -------------------------
    # Load Model
    # -------------------------
    model = LightningCNN.load_from_checkpoint("models/cnn_lightning.ckpt")
    model.eval()

    # -------------------------
    # Run Evaluation
    # -------------------------
    y_true, y_pred, y_prob = [], [], []

    for x, y in test_loader:
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()

        y_true.extend(y.numpy())
        y_pred.extend(preds.numpy())
        y_prob.extend(probs.numpy())

    # -------------------------
    # Compute Metrics
    # -------------------------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # -------------------------
    # Save Plots
    # -------------------------
    os.makedirs("plots", exist_ok=True)

    plot_confusion_matrix(y_true, y_pred, save_path="plots/confusion_matrix.png")
    plot_roc_curve(y_true, y_prob, model_name="CNN", save_path="plots/roc_curve.png")
    plot_precision_recall(
        y_true, y_prob, model_name="CNN", save_path="plots/pr_curve.png"
    )


if __name__ == "__main__":  # 🔹 Required on Windows for multiprocessing
    evaluate_model()
