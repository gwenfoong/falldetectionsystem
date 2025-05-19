import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt

from pipeline.utils import (
    get_device,
    load_temporal_model
)

def evaluate_temporal_model(
    model_path: str,
    data_root: str,
    seq_len: int = 16,
    output_dir: str = None,
    plot_curves: bool = True
):
    """
    Runs sequence-based inference over each folder of exactly `seq_len` frames.
    Sweeps thresholds, prints report, and optionally plots ROC/PR.
    If output_dir is provided, ensures it exists (for saving any outputs).
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    model  = load_temporal_model(model_path)

    # frame transform
    frame_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    def load_seq(folder):
        imgs = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])
        if len(imgs) < seq_len:
            return None
        last = imgs[-seq_len:]
        ts = [frame_transform(Image.open(os.path.join(folder,fn)).convert("RGB"))
              for fn in last]
        return torch.stack(ts, dim=0).unsqueeze(0).to(device)  # [1,seq,C,H,W]

    y_true, y_scores = [], []
    for cat, label in [("falls",1), ("non-falls",0)]:
        cat_dir = os.path.join(data_root, cat)
        if not os.path.isdir(cat_dir):
            continue
        for seq in os.listdir(cat_dir):
            seq_folder = os.path.join(cat_dir, seq)
            if not os.path.isdir(seq_folder):
                continue
            seq_tensor = load_seq(seq_folder)
            if seq_tensor is None:
                continue
            with torch.no_grad():
                logits = model(seq_tensor)
                prob   = F.softmax(logits, dim=1)[0,1].item()
            y_true.append(label)
            y_scores.append(prob)
            print(f"{cat}/{seq}: prob={prob:.3f}")

    y_true   = np.array(y_true)
    y_scores = np.array(y_scores)

    # threshold sweep
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1,0.9,17):
        preds = (y_scores >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, preds, average='binary', zero_division=0
        )
        print(f"Thresh={t:.2f}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"\n>> Best temporal threshold: {best_t:.2f} (F1={best_f1:.3f})\n")

    # final report
    final_preds = (y_scores >= best_t).astype(int)
    print(classification_report(
        y_true, final_preds,
        target_names=["non-fall","fall"],
        zero_division=0
    ))
    print("Confusion Matrix:\n", confusion_matrix(y_true, final_preds))

    # optional curves
    if plot_curves:
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc     = auc(fpr, tpr)
        plt.figure(); plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.2f}")
        plt.plot([0,1],[0,1],"--"); plt.title("Temporal ROC")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.show()

        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        pr_auc       = auc(rec, prec)
        plt.figure(); plt.plot(rec, prec, label=f"PR AUC={pr_auc:.2f}")
        plt.title("Temporal Precision-Recall")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.show()
