import os
import cv2
import numpy as np
import torch
from collections import deque
from PIL import Image
from pipeline.utils import preprocess_frame, get_device
from pipeline.models.my_model import MyBaselineModel

# ── Configuration ──────────────────────────────────────────────────────────────
YOLO_CFG          = "/Users/gwen/ITSS_Project/darknet/cfg/yolov3.cfg"
YOLO_WEIGHTS      = "/Users/gwen/ITSS_Project/yolov3.weights"
YOLO_NAMES        = "/Users/gwen/ITSS_Project/darknet/data/coco.names"
VIDEO_SOURCE      = "/Users/gwen/ITSS_Project/processed_data2/CAUCAFall/falls/FallBackwardssourvS2.mp4"
OUTPUT_DIR        = "/Users/gwen/ITSS_Project/sample_video2"
FRAME_MODEL_PATH  = "/Users/gwen/ITSS_Project/models/baseline_model_frame.pth"
CONF_THRESH       = 0.5
NMS_THRESH        = 0.4
DETECT_INTERVAL   = 10

# ── Load YOLO ─────────────────────────────────────────────────────────────────
with open(YOLO_NAMES, 'r') as f:
    classes = [c.strip() for c in f]
net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
ln = net.getLayerNames()
out_ln = [ln[i-1] for i in net.getUnconnectedOutLayers().flatten()]

def detect_persons(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_ln)
    boxes, confs = [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            cid = np.argmax(scores)
            conf = float(scores[cid])
            if classes[cid]=="person" and conf>CONF_THRESH:
                cx, cy = int(det[0]*w), int(det[1]*h)
                bw, bh = int(det[2]*w), int(det[3]*h)
                x, y = cx-bw//2, cy-bh//2
                boxes.append([x,y,bw,bh])
                confs.append(conf)
    idxs = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESH, NMS_THRESH)
    return [boxes[i] for i in idxs.flatten()] if len(idxs)>0 else []

# ── Load fall model ────────────────────────────────────────────────────────────
device     = get_device()
fall_model = MyBaselineModel(num_classes=2).to(device)
sd         = torch.load(FRAME_MODEL_PATH, map_location=device)
# strip any "model." prefix if present
sd = {k.replace("model.",""):v for k,v in sd.items()}
fall_model.load_state_dict(sd, strict=False)
fall_model.eval()

# ── Video I/O ─────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
cap       = cv2.VideoCapture(VIDEO_SOURCE)
base      = os.path.splitext(os.path.basename(VIDEO_SOURCE))[0]
out_path  = os.path.join(OUTPUT_DIR, f"output_with_tracking_{base}.mp4")
fourcc    = cv2.VideoWriter_fourcc(*'mp4v')
writer    = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                            (int(cap.get(3)),int(cap.get(4))))

tracker    = None
last_box   = None
frame_count= 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    if frame_count % DETECT_INTERVAL == 1 or tracker is None:
        dets = detect_persons(frame)
        if dets:
            box = max(dets, key=lambda b: b[2]*b[3])
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, tuple(box))
            last_box = box

    if tracker:
        ok, box = tracker.update(frame)
        if ok:
            last_box = box
        x,y,w,h = [int(v) for v in last_box]
        roi     = frame[y:y+h, x:x+w]
        label   = "Fall" if preprocess_frame(roi).to(device).pipe(
                      lambda t: torch.softmax(fall_model(t),1)[0,1].item()
                  ) > 0.5 else "Person"
        color   = (0,0,255) if label=="Fall" else (0,255,0)
        cv2.rectangle(frame, (x,y),(x+w,y+h), color, 2)
        cv2.putText(frame, label, (x, max(10,y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    writer.write(frame)

cap.release()
writer.release()
print(f"Saved: {out_path}")
