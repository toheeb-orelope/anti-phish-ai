import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

def plot_confusion_matrix(y_true, y_pred, labels=["Benign", "Phish"], save_path=None):
    """Plot and optionally save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[+] Saved confusion matrix → {save_path}")
    plt.show()

def plot_roc_curve(y_true, y_prob, model_name="Model", save_path=None):
    """Plot and optionally save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[+] Saved ROC curve → {save_path}")
    plt.show()

def plot_precision_recall(y_true, y_prob, model_name="Model", save_path=None):
    """Plot and optionally save Precision-Recall curve."""
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(rec, prec, label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[+] Saved Precision-Recall curve → {save_path}")
    plt.show()
