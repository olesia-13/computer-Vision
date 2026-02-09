import os
import cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, 'traffic_video')
OUT_DIR = os.path.join(PROJECT_DIR, 'out')

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

USE_WEBCAM = False

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    VIDEO_PATH = os.path.join(VIDEO_DIR, "cars.mp4")
    cap = cv2.VideoCapture(VIDEO_PATH)

input_fps = cap.get(cv2.CAP_PROP_FPS)
if input_fps == 0:
    input_fps = 25

model = YOLO('yolov8n.pt')

CONF_TRESHOLD = 0.4
RESIZE_WIDTH = 960

prev_time = time.time()
fps = 0.0


VEHICLE_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


writer = None
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))


    if writer is None:
        out_path = os.path.join(OUT_DIR, "result.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, input_fps, (frame.shape[1], frame.shape[0]))

    result = model(frame, conf=CONF_TRESHOLD, verbose=False)


    counts = {name: 0 for name in VEHICLE_CLASSES.values()}

    for r in result:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])


            if cls not in VEHICLE_CLASSES:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label_name = VEHICLE_CLASSES[cls]
            counts[label_name] += 1

            color = (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{label_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)


    now = time.time()
    dt = now - prev_time
    prev_time = now
    if dt > 0:
        fps = 1.0 / dt


    y = 30
    total = 0
    for name, value in counts.items():
        cv2.putText(frame, f"{name}: {value}", (20, y), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 2)
        y += 30
        total += value

    cv2.putText(frame, f"Total: {total}", (20, y), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,255), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y+30), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,255,255), 2)

    writer.write(frame)
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
