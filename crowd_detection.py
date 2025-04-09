import cv2
import torch
import numpy as np
import sys
from deep_sort_realtime.deepsort_tracker import DeepSort

sys.path.insert(0, "D://codes//yolov5")
from yolov5.utils.torch_utils import select_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load YOLOv5 model (pretrained)
model = torch.hub.load('yolov5', 'yolov5m', source='local', pretrained=True).to(device)
model.conf = 0.25
model.classes = [0]  # Detect only 'person'
model.augment = True
model.eval()
model.iou = 0.35

# Accept video path or CCTV stream URL from command-line
if len(sys.argv) < 2:
    print("Usage: python crowd_detection.py <video_path_or_stream_url>")
    sys.exit(1)

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Error: Cannot open video source: {video_path}")
    sys.exit(1)

# Initialize DeepSort tracker
tracker = DeepSort(max_age=25, n_init=2, nn_budget=100)

# Tracking variables
previous_tracks = []
entered_ids = set()
exited_ids = set()

# IOU calculation
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

count_mode = "live"

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ End of video stream or cannot fetch frame.")
        break

    frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    detections = results.pred[0]

    person_detections = []
    for *box, conf, cls in detections:
        if int(cls) == 0:
            x1, y1, x2, y2 = map(int, box)
            width = x2 - x1
            height = y2 - y1
            if conf.item() >= 0.4 and width > 20 and height > 40:
                person_detections.append(([x1, y1, width, height], conf.item(), 'person'))

    filtered_detections = []

    if previous_tracks:
        for det in person_detections:
            box, conf, label = det
            keep = False
            for prev in previous_tracks:
                prev_box = prev['bbox']
                if iou(box, prev_box) > 0.2:
                    keep = True
                    break
            if keep:
                filtered_detections.append(det)
    else:
        filtered_detections = person_detections.copy()

    tracks = tracker.update_tracks(filtered_detections, frame=frame)

    previous_tracks = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        l, t, r, b = track.to_ltrb()
        previous_tracks.append({'id': track.track_id, 'bbox': [l, t, r - l, b - t]})

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_tlbr()
        x_center = int((bbox[0] + bbox[2]) / 2)
        y_center = int((bbox[1] + bbox[3]) / 2)
        center = (x_center, y_center)

        x1, y1, x2, y2 = map(int, track.to_tlbr())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

    if count_mode == "live":
        live_count = sum(1 for track in tracks if track.is_confirmed())
        cv2.putText(frame, f"People on screen: {live_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Tracking People", frame)

    key = cv2.waitKey(1) & 0xFF
    if cv2.getWindowProperty("Tracking People", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
