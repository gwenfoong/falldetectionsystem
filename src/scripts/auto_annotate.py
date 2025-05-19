#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
from PIL import Image
from pipeline.utils import preprocess_frame, preprocess_sequence, get_device
from pipeline.models.my_model       import MyBaselineModel
from pipeline.models.temporal_model import MyTemporalModel
from collections import deque

# ---------------------------
# Configuration
# ---------------------------
DATASET_ROOTS = {
    "GMDCSA24": "/Users/gwen/ITSS_Project/processed_data/GMDCSA24"
}
CATEGORIES           = ["falls","non-falls"]
FRAME_MODEL_PATH     = "/Users/gwen/ITSS_Project/models/baseline_model_frame.pth"
TEMPORAL_MODEL_PATH  = "/Users/gwen/ITSS_Project/models/baseline_model_temporal.pth"
THRESHOLD            = 0.4
SEQ_LENGTH           = 16

# ---------------------------
# Helpers
# ---------------------------
def group_fall_segments(probs, files):
    segments, in_seg, seg_probs, start = [], False, [], None
    for i,p in enumerate(probs):
        if p >= THRESHOLD:
            if not in_seg:
                in_seg, start, seg_probs = True, i, [p]
            else:
                seg_probs.append(p)
        else:
            if in_seg:
                segments.append({
                    "start_frame": files[start],
                    "end_frame":   files[i-1],
                    "average_probability": float(np.mean(seg_probs))
                })
                in_seg = False
    if in_seg:
        segments.append({
            "start_frame": files[start],
            "end_frame":   files[-1],
            "average_probability": float(np.mean(seg_probs))
        })
    return segments

def annotate_folder(model, path, is_temporal, out_dir):
    frame_files = sorted(f for f in os.listdir(path)
                         if f.lower().endswith((".jpg",".png")))
    if not frame_files: return []
    device = get_device()
    annots, window = [], deque(maxlen=SEQ_LENGTH)

    if not is_temporal:
        # frame-only
        for f in frame_files:
            img = Image.open(os.path.join(path,f)).convert("RGB")
            inp = preprocess_frame(np.array(img)).to(device)
            prob = float(torch.softmax(model(inp),1)[0,1].item())
            annots.append((f, prob))
        return group_fall_segments([p for _,p in annots], [f for f,_ in annots])

    # temporal
    # slide
    for f in frame_files:
        img = Image.open(os.path.join(path,f)).convert("RGB")
        window.append(np.array(img))
        if len(window) == SEQ_LENGTH:
            inp = preprocess_sequence(list(window)).to(device)
            prob = float(torch.softmax(model(inp),1)[0,1].item())
            annots.append((f,prob))
    if not annots: return []
    return group_fall_segments([p for _,p in annots], [f for f,_ in annots])

# ---------------------------
# Main
# ---------------------------
def main():
    # load models
    device = get_device()
    frame_model    = MyBaselineModel().to(device)
    frame_model.load_state_dict(torch.load(FRAME_MODEL_PATH,map_location=device))
    frame_model.eval()

    temp_model     = MyTemporalModel().to(device)
    temp_model.load_state_dict(torch.load(TEMPORAL_MODEL_PATH,map_location=device))
    temp_model.eval()

    for ds_name, ds_root in DATASET_ROOTS.items():
        for cat in CATEGORIES:
            cat_folder = os.path.join(ds_root, cat)
            if not os.path.isdir(cat_folder): continue

            out_base = os.path.join(ds_root, "auto_annotations", cat)
            os.makedirs(out_base, exist_ok=True)

            for vid in os.listdir(cat_folder):
                vid_path = os.path.join(cat_folder, vid)
                if not os.path.isdir(vid_path): continue

                # frame-level annotations
                segs_f = annotate_folder(frame_model, vid_path, False, out_base)
                json.dump({
                    "video": vid,
                    "fall_segments": segs_f,
                    "model":"frame"
                }, open(os.path.join(out_base, vid+".json"),"w"), indent=2)

                # temporal-level annotations
                segs_t = annotate_folder(temp_model, vid_path, True, out_base)
                json.dump({
                    "video": vid,
                    "fall_segments": segs_t,
                    "model":"temporal"
                }, open(os.path.join(out_base, vid+"_temp.json"),"w"), indent=2)

if __name__=="__main__":
    main()
