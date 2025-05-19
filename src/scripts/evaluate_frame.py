import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
from pipeline.utils import load_frame_model, get_device, preprocess_frame
from pipeline.models.my_model import MyBaselineModel

# ── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT       = "/Users/gwen/ITSS_Project/processed_data/CAUCAFall"
FRAME_MODEL_PTH = "/Users/gwen/ITSS_Project/models/baseline_model_frame.pth"
device          = get_device()

def evaluate():
    model = load_frame_model(FRAME_MODEL_PTH)
    y_true, y_scores = [], []

    for cat, label in [("falls",1),("non-falls",0)]:
        cat_dir = os.path.join(DATA_ROOT, cat)
        for vid in os.listdir(cat_dir):
            folder = os.path.join(cat_dir, vid)
            if not os.path.isdir(folder): continue
            frames = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))
            if not frames: continue
            img = preprocess_frame(plt.imread(os.path.join(folder,frames[-1])))
            prob = float(F.softmax(model(img.to(device)),1)[0,1].item())
            y_true.append(label); y_scores.append(prob)
            print(f"{cat}/{vid}: {prob:.3f}")

    y_true, y_scores = np.array(y_true), np.array(y_scores)

    best_t, best_f1 = 0.5, 0
    for t in np.linspace(0.1,0.9,17):
        preds = (y_scores>=t).astype(int)
        p,r,f1,_ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0
        )
        print(f"Thresh={t:.2f}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
        if f1>best_f1: best_f1, best_t = f1, t

    print(f"\n>> Best thresh {best_t:.2f}, F1 {best_f1:.3f}\n")
    final_preds = (y_scores>=best_t).astype(int)
    print(classification_report(y_true, final_preds, target_names=["non-fall","fall"], zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_true, final_preds))

    # ROC
    fpr,tpr,_ = roc_curve(y_true,y_scores)
    plt.figure(); plt.plot(fpr,tpr,label=f"AUC={auc(fpr,tpr):.2f}")
    plt.plot([0,1],[0,1],"--"); plt.title("ROC"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.show()

    # PR
    prec,rec,_ = precision_recall_curve(y_true,y_scores)
    plt.figure(); plt.plot(rec,prec,label=f"AUC={auc(rec,prec):.2f}")
    plt.title("Precision-Recall"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.show()

if __name__=="__main__":
    evaluate()
