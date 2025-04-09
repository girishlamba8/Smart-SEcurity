import os
import json
import cv2
import torch
import numpy as np
import sys
from deep_sort_realtime.deepsort_tracker import DeepSort

sys.path.insert(0, "D://codes//yolov5")
from yolov5.utils.torch_utils import select_device

polygon_file = "saved_polygons.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = torch.hub.load('yolov5', 'yolov5m', source='local', pretrained=True).to(device)
model.conf = 0.7
model.classes = [0]  # Only detect person
model.eval()

# Accept either CCTV stream link or video file path
input_source = sys.argv[1] if len(sys.argv) > 1 else 0  # Default to webcam if nothing passed
cap = cv2.VideoCapture(input_source)

tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

drawing = False
current_polygon = []
polygons = []
zone_complete = False
inside_by_zone = []

def mouse_callback(event, x, y, flags, param):
    global current_polygon, polygons, zone_complete
    if zone_complete:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_polygon) >= 3:
            polygons.append(current_polygon.copy())
            inside_by_zone.append(set())
            current_polygon.clear()
            print(f"Zone {len(polygons)} completed.")
            with open(polygon_file, "w") as f:
                json.dump(polygons, f)
        else:
            print("Need at least 3 points to complete a zone.")

cv2.namedWindow("Work Monitoring")
cv2.setMouseCallback("Work Monitoring", mouse_callback)

# Load saved polygons if available
if os.path.exists(polygon_file) and os.path.getsize(polygon_file) > 0:
    try:
        with open(polygon_file, "r") as f:
            polygons = json.load(f)
        inside_by_zone = [set() for _ in polygons]
        print(f"Loaded {len(polygons)} saved zone(s) from file.")
    except json.JSONDecodeError:
        print("Warning: Polygon file is corrupted or not valid JSON. Starting fresh.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    detections = results.pred[0]

    person_detections = []
    for *box, conf, cls in detections:
        if int(cls) == 0:
            x1, y1, x2, y2 = map(int, box)
            person_detections.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), 'person'))

    tracks = tracker.update_tracks(person_detections, frame=frame)

    overlay = frame.copy()

    if current_polygon:
        for pt in current_polygon:
            cv2.circle(overlay, pt, 4, (0, 255, 255), -1)
        if len(current_polygon) > 1:
            cv2.polylines(overlay, [np.array(current_polygon, np.int32)], False, (255, 255, 0), 1)

    for i, poly in enumerate(polygons):
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], (0, 200, 255))
        cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)
        cv2.putText(overlay, f"Zone {i+1}", pts[0][0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    for i in range(len(inside_by_zone)):
        inside_by_zone[i].clear()

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        for i, poly in enumerate(polygons):
            if cv2.pointPolygonTest(np.array(poly, np.int32), center, False) >= 0:
                inside_by_zone[i].add(track_id)

    for i, ids in enumerate(inside_by_zone):
        if len(ids) == 0:
            cv2.putText(frame, f" Zone {i+1} empty!", (20, 60 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #print(f"ALERT: Zone {i+1} is empty!")

    cv2.imshow("Work Monitoring", frame)

    key = cv2.waitKey(1) & 0xFF
    if cv2.getWindowProperty("Work Monitoring", cv2.WND_PROP_VISIBLE) < 1:
        break
    elif key == ord('r'):
        polygons.clear()
        current_polygon.clear()
        inside_by_zone.clear()
        if os.path.exists(polygon_file):
            os.remove(polygon_file)
            print("Polygon file removed.")
        print("All zones cleared. Draw again.")
    elif key == ord('n'):
        if len(current_polygon) >= 3:
            polygons.append(current_polygon.copy())
            inside_by_zone.append(set())
            with open(polygon_file, "w") as f:
                json.dump(polygons, f)
            current_polygon.clear()
            print(f"Zone {len(polygons)} completed.")
        else:
            print("Need at least 3 points to complete a zone.")

cap.release()
cv2.destroyAllWindows()
