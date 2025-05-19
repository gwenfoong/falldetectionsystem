import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from pipeline.utils import (
    get_device,
    load_frame_model
)

# ── Configuration ─────────────────────────────────────────────────────────────
def evaluate_frame_model(
    model_path: str,
    data_root: str,
    output_dir: str,
    plot_curves: bool = True
):
    """
    Runs frame-only inference over last frame of each video folder.
    Sweeps thresholds to pick best F1, prints report, and optionally plots ROC/PR.
    """
    device = get_device()
    model  = load_frame_model(model_path)

    # single-image transform
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
    def load_image(path):
        return transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

    y_true, y_scores = [], []
    for cat, label in [("falls",1), ("non-falls",0)]:
        cat_dir = os.path.join(data_root, cat)
        if not os.path.isdir(cat_dir):
            continue
        for vid in os.listdir(cat_dir):
            vid_folder = os.path.join(cat_dir, vid)
            if not os.path.isdir(vid_folder):
                continue
            frames = sorted(f for f in os.listdir(vid_folder)
                            if f.lower().endswith((".jpg",".png")))
            if not frames:
                continue
            img   = load_image(os.path.join(vid_folder, frames[-1]))
            with torch.no_grad():
                logit = model(img)
                prob  = F.softmax(logit, dim=1)[0,1].item()
            y_true.append(label)
            y_scores.append(prob)

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

    print(f"\n>> Best frame threshold: {best_t:.2f} (F1={best_f1:.3f})\n")
    final_preds = (y_scores >= best_t).astype(int)
    print(classification_report(
        y_true, final_preds,
        target_names=["non-fall","fall"],
        zero_division=0
    ))
    print("Confusion Matrix:\n", confusion_matrix(y_true, final_preds))

    if plot_curves:
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc     = auc(fpr, tpr)
        plt.figure(); plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.2f}")
        plt.plot([0,1],[0,1],"--"); plt.title("Frame ROC")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.show()

        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(rec, prec)
        plt.figure(); plt.plot(rec, prec, label=f"PR AUC={pr_auc:.2f}")
        plt.title("Frame Precision-Recall")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.show()
