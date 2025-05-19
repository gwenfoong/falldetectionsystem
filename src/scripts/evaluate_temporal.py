import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from pipeline.utils import get_device
from pipeline.models.temporal_model import MyTemporalModel
from PIL import Image
from torchvision import transforms

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET_ROOT        = "/Users/gwen/ITSS_Project/processed_data/CAUCAFall"
MODEL_PATH          = "/Users/gwen/ITSS_Project/models/baseline_model_temporal.pth"
SEQ_LENGTH          = 16

# ── Device & model ─────────────────────────────────────────────────────────────
device = get_device()
model  = MyTemporalModel(num_classes=2, hidden_size=256).to(device)
ckpt   = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt, strict=False)
model.eval()

# ── Frame transform ────────────────────────────────────────────────────────────
frame_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

def load_sequence(folder_path):
    imgs = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".jpg"))
    if len(imgs) < SEQ_LENGTH:
        return None
    last_imgs = imgs[-SEQ_LENGTH:]
    tensors  = []
    for fn in last_imgs:
        img = Image.open(os.path.join(folder_path, fn)).convert("RGB")
        tensors.append(frame_transform(img))
    return torch.stack(tensors, dim=0).unsqueeze(0).to(device)  # [1, S, C, H, W]

# ── Collect scores ─────────────────────────────────────────────────────────────
y_true, y_scores = [], []
for category, label in [("falls",1), ("non-falls",0)]:
    cat_dir = os.path.join(DATASET_ROOT, category)
    if not os.path.isdir(cat_dir):
        continue
    for seq in os.listdir(cat_dir):
        path = os.path.join(cat_dir, seq)
        if not os.path.isdir(path):
            continue
        seq_tensor = load_sequence(path)
        if seq_tensor is None:
            continue
        with torch.no_grad():
            logits = model(seq_tensor)
            prob   = F.softmax(logits, dim=1)[0,1].item()
        y_true.append(label)
        y_scores.append(prob)
        print(f"{category}/{seq}: prob={prob:.3f}")

y_true   = np.array(y_true)
y_scores = np.array(y_scores)

# ── Threshold sweep ────────────────────────────────────────────────────────────
best_t, best_f1 = 0.5, 0.0
for t in np.linspace(0.1,0.9,17):
    preds = (y_scores >= t).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, preds, average='binary', zero_division=0
    )
    print(f"Thresh={t:.2f}: P={p:.3f}, R={r:.3f}, F1={f
