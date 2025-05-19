import os, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from pipeline.models.my_model       import MyBaselineModel
from pipeline.models.temporal_model import MyTemporalModel
from pipeline.utils import get_device
from sklearn.metrics import classification_report

# ── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT            = "/Users/gwen/ITSS_Project/processed_data/CAUCAFall"
FRAME_SAVE_PATH      = "/Users/gwen/ITSS_Project/models/baseline_model_frame.pth"
TEMP_SAVE_PATH       = "/Users/gwen/ITSS_Project/models/baseline_model_temporal.pth"
BATCH_SIZE, EPOCHS   = 32, 10
LR, SPLIT, SEQ_LEN   = 1e-4, 0.8, 16
DEVICE               = get_device()

# ── Datasets ─────────────────────────────────────────────────────────────────
# assumes processed_data/CAUCAFall/falls/* and non-falls/* each contain subfolders
from pipeline.scripts.datasets import CombinedFrameDataset, CombinedTemporalDataset

frame_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

def train_model(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, n = 0,0,0
    for X,y in loader:
        X,y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(X)
        loss=criterion(out,y)
        loss.backward(); optimizer.step()
        total_loss += loss.item()*X.size(0)
        _,preds = out.max(1)
        correct += (preds==y).sum().item()
        n += X.size(0)
    return total_loss/n, correct/n

def eval_model(model, loader):
    model.eval()
    all_preds, all_labels = [],[]
    with torch.no_grad():
        for X,y in loader:
            X,y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            _,preds = out.max(1)
            all_preds += preds.cpu().tolist()
            all_labels += y.cpu().tolist()
    acc = sum(p==t for p,t in zip(all_preds,all_labels))/len(all_labels)
    return acc, all_labels, all_preds

# ── Training loops ───────────────────────────────────────────────────────────
def main():
    print("=== Training frame baseline ===")
    full = CombinedFrameDataset(DATA_ROOT, frame_transform)
    n_train = int(len(full)*SPLIT); n_val=len(full)-n_train
    train_ds, val_ds = random_split(full,[n_train,n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    frame_model = MyBaselineModel().to(DEVICE)
    opt_f = optim.Adam(frame_model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    best_acc=0
    for ep in range(EPOCHS):
        tr_loss, tr_acc = train_model(frame_model, train_loader, crit, opt_f)
        val_acc,_,_     = eval_model(frame_model, val_loader)
        print(f"Epoch {ep+1}: train={tr_acc:.3f}, val={val_acc:.3f}")
        if val_acc>best_acc:
            best_acc=val_acc
            torch.save(frame_model.state_dict(), FRAME_SAVE_PATH)
    print("Frame eval report:", classification_report(
        val_ds.dataset.labels, 
        eval_model(frame_model, val_loader)[2],
        target_names=["non-fall","fall"]
    ))

    print("=== Training temporal baseline ===")
    temp_ds = CombinedTemporalDataset(DATA_ROOT, SEQ_LEN, frame_transform)
    n_train = int(len(temp_ds)*SPLIT); n_val=len(temp_ds)-n_train
    train_t,val_t = random_split(temp_ds,[n_train,n_val],
                                 generator=torch.Generator().manual_seed(42))
    t_loader,v_loader = DataLoader(train_t,batch_size=BATCH_SIZE),DataLoader(val_t,batch_size=BATCH_SIZE)

    temp_model = MyTemporalModel().to(DEVICE)
    opt_t = optim.Adam(temp_model.parameters(), lr=LR)
    best_acc=0
    for ep in range(EPOCHS):
        tr_loss, tr_acc = train_model(temp_model, t_loader, crit, opt_t)
        val_acc,_,_     = eval_model(temp_model, v_loader)
        print(f"Epoch {ep+1}: train={tr_acc:.3f}, val={val_acc:.3f}")
        if val_acc>best_acc:
            best_acc=val_acc
            torch.save(temp_model.state_dict(), TEMP_SAVE_PATH)
    print("Temporal eval report:", classification_report(
        val_t.dataset.labels, 
        eval_model(temp_model, v_loader)[2],
        target_names=["non-fall","fall"]
    ))

if __name__=="__main__":
    main()
