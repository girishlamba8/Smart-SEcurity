import cv2
import numpy as np
import math
import sys
import time

# Initialize people detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Accept either a CCTV stream URL or a local video file
input_source = sys.argv[1] if len(sys.argv) > 1 else 0  # Use webcam if nothing is passed
cap = cv2.VideoCapture(input_source)

def get_center(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Time variables
fight_display_duration = 4  # seconds
last_fight_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8))
    centers = [get_center(x, y, w, h) for (x, y, w, h) in boxes]

    fight_detected = False
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = euclidean_distance(centers[i], centers[j])
            if dist < 50:  # Threshold for possible fight proximity
                fight_detected = True
                cv2.line(frame, centers[i], centers[j], (0, 0, 255), 2)

    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if fight_detected:
        last_fight_time = time.time()

    # Show the message if it's within the display window
    if time.time() - last_fight_time < fight_display_duration:
        cv2.putText(frame, "Possible Fight Detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Fight Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if cv2.getWindowProperty("Fight Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()

