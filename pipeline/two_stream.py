import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import yaml
from tqdm import tqdm

from pipeline.utils import (
    get_device,
    preprocess_frame,
    preprocess_sequence,
    load_frame_model,
    load_temporal_model,
)

# ─── YOLO + tracking params ────────────────────────────────────────────────────
YOLO_CFG      = "darknet/cfg/yolov3.cfg"
YOLO_WEIGHTS  = "yolov3.weights"
YOLO_NAMES    = "darknet/data/coco.names"
CONF_THRESH   = 0.5
NMS_THRESH    = 0.4
DETECT_EVERY  = 10    # run YOLO every N frames
MIN_ROI       = 64    # min half-side of the square crop

def detect_persons(frame, net, out_names):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_names)

    boxes, confs = [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            cid    = int(np.argmax(scores))
            conf   = float(scores[cid])
            if conf < CONF_THRESH or classes[cid] != "person":
                continue
            bx, by, bw, bh = det[0]*w, det[1]*h, det[2]*w, det[3]*h
            x = int(bx - bw/2); y = int(by - bh/2)
            boxes.append([x, y, int(bw), int(bh)])
            confs.append(conf)

    idxs = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESH, NMS_THRESH)
    if len(idxs) == 0:
        return []
    idxs = np.array(idxs).flatten().tolist()
    return [boxes[i] for i in idxs if 0 <= i < len(boxes)]


def run_two_stream(
    frame_model_path: str,
    temporal_model_path: str,
    input_video: str,
    output_dir: str
):
    # 1) load config
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__),"config.yaml")))
    ALPHA      = cfg["fusion_alpha"]
    THRESHOLD  = cfg["fusion_threshold"]
    SEQ_LEN    = cfg["sequence_length"]
    SMOOTH_WIN = cfg["smoothing_window"]

    os.makedirs(output_dir, exist_ok=True)
    device = get_device()

    # 2) load your two nets
    fm = load_frame_model(frame_model_path)
    tm = load_temporal_model(temporal_model_path)

    # 3) init YOLO + class names
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    ln  = net.getLayerNames()
    out_names = [ln[i-1] for i in net.getUnconnectedOutLayers().flatten()]
    with open(YOLO_NAMES) as f:
        global classes
        classes = [c.strip() for c in f]

    # 4) open video I/O
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_video}")
    fps    = cap.get(cv2.CAP_PROP_FPS)
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(output_dir, "two_stream.mp4")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    # 5) prepare tracker & buffers
    tracker    = None
    last_box   = None
    seq_buf    = deque(maxlen=SEQ_LEN)
    smooth_buf = deque(maxlen=SMOOTH_WIN)

    # 6) main loop with tqdm
    with tqdm(total=total, desc="Two-Stream Inference") as pbar:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # re-detect every DETECT_EVERY frames or if tracker lost
            if frame_idx % DETECT_EVERY == 1 or tracker is None:
                dets = detect_persons(frame, net, out_names)
                if dets:
                    box = max(dets, key=lambda b: b[2]*b[3])
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, tuple(box))
                    last_box = box

            if tracker:
                ok, box = tracker.update(frame)
                if ok:
                    last_box = box
                x,y,w_box,h_box = [int(v) for v in last_box]

                # crop a square ROI around the person
                cx, cy = x + w_box//2, y + h_box//2
                half   = max(w_box, h_box, MIN_ROI)//2
                x1, y1 = max(0, cx-half), max(0, cy-half)
                x2, y2 = min(W, cx+half), min(H, cy+half)
                roi = frame[y1:y2, x1:x2]

                # frame-level score
                ft = preprocess_frame(roi).to(device)
                with torch.no_grad():
                    pf = F.softmax(fm(ft), dim=1)[0,1].item()

                # temporal score once we have SEQ_LEN frames
                seq_buf.append(roi)
                if len(seq_buf) == SEQ_LEN:
                    st = preprocess_sequence(list(seq_buf)).to(device)
                    with torch.no_grad():
                        pt = F.softmax(tm(st), dim=1)[0,1].item()
                else:
                    pt = pf

                # fuse + push into median buffer
                fused = ALPHA * pf + (1 - ALPHA) * pt
                smooth_buf.append(fused)
                score = float(np.median(smooth_buf))

                # *** warm-up: no “Fall” until buffer is full ***
                is_fall = False
                if len(smooth_buf) == SMOOTH_WIN and score >= THRESHOLD:
                    is_fall = True

                label = f"Fall ({score:.2f})" if is_fall else f"No Fall ({score:.2f})"
                cv2.putText(
                    frame, label, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2
                )

            writer.write(frame)
            pbar.update(1)

    cap.release()
    writer.release()
    print(f"Saved two-stream demo to {out_path}")
