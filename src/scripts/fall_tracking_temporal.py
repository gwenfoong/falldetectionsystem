import os
import cv2
import numpy as np
import torch
from collections import deque
from pipeline.models.temporal_model import MyTemporalModel
from pipeline.utils import preprocess_sequence, get_device
from torchvision.models import ResNet50_Weights
from PIL import Image
import torchvision.transforms as transforms

# ── Configuration ──────────────────────────────────────────────────────────────
YOLO_CFG      = "/Users/gwen/ITSS_Project/darknet/cfg/yolov3.cfg"
YOLO_WEIGHTS  = "/Users/gwen/ITSS_Project/yolov3.weights"
YOLO_NAMES    = "/Users/gwen/ITSS_Project/darknet/data/coco.names"
VIDEO_SOURCE  = "/Users/gwen/ITSS_Project/processed_data/CAUCAFall/falls/FallBackwardsS6.mp4"
MODEL_PATH    = "/Users/gwen/ITSS_Project/models/baseline_model_temporal.pth"
OUTPUT_DIR    = "/Users/gwen/ITSS_Project/sample_video2"
SEQ_LEN       = 10
PRED_THRESH   = 0.3
CONF_THRESH   = 0.5
NMS_THRESH    = 0.4
DETECT_INT    = 10
SMOOTH_WIN    = 3
CROP_SIZE     = 300
MIN_ROI       = 150

# ── Load YOLO person detector ──────────────────────────────────────────────────
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
                boxes.append([x,y,bw,bh]); confs.append(conf)
    idxs = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESH, NMS_THRESH)
    return [boxes[i] for i in idxs.flatten()] if len(idxs)>0 else []

# ── Load temporal model ────────────────────────────────────────────────────────
device = get_device()
model  = MyTemporalModel(num_classes=2, hidden_size=256).to(device)
ckpt   = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt, strict=False)
model.eval()

# ── Transform for ROI ──────────────────────────────────────────────────────────
roi_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ── Run inference & tracking ───────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_SOURCE)
base = os.path.splitext(os.path.basename(VIDEO_SOURCE))[0]
out_path = os.path.join(OUTPUT_DIR, f"output_with_tracking_{base}.mp4")
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                         cap.get(cv2.CAP_PROP_FPS),
                         (int(cap.get(3)), int(cap.get(4))))

tracker    = None
last_box   = None
frame_seq  = deque(maxlen=SEQ_LEN)
smooth_buf = deque(maxlen=SMOOTH_WIN)
frame_count= 0

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1

    # redetect every DETECT_INT frames
    if frame_count % DETECT_INT == 1 or tracker is None:
        dets = detect_persons(frame)
        if dets:
            box = max(dets, key=lambda b: b[2]*b[3])
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, tuple(box))
            last_box = box

    # update tracker
    if tracker:
        ok, box = tracker.update(frame)
        if ok:
            last_box = box
        x,y,w,h = [int(v) for v in last_box]
        cx, cy = x+w//2, y+h//2
        half = CROP_SIZE//2
        x1, y1 = max(0, cx-half), max(0, cy-half)
        x2, y2 = min(frame.shape[1], cx+half), min(frame.shape[0], cy+half)
        roi = frame[y1:y2, x1:x2]
        if roi.shape[0]>=MIN_ROI and roi.shape[1]>=MIN_ROI:
            frame_seq.append(roi)

        # when we have a full sequence, predict
        if len(frame_seq)==SEQ_LEN:
            seq_tensor = preprocess_sequence(list(frame_seq_
