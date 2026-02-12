import cv2
import os
import csv
import time
from ultralytics import YOLO
import yt_dlp

# --- НАЛАШТУВАННЯ ---
YOUTUBE_URL = "https://www.youtube.com/live/Lxqcg1qt0XU?si=ZITYP0db8lhigDag"
USE_YOUTUBE = True
SKIP_FRAMES = 2

line1 = [(400, 640), (1300, 780)]
line2 = [(900, 480), (1600, 550)]
DISTANCE_METERS = 25

# Класи COCO: 2: car, 3: motorcycle, 5: bus, 7: truck
ALLOWED_CLASSES = [2, 3, 5, 7]

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, 'traffic_data.csv')

def get_stream_url(url):
    ydl_opts = {'format': 'best', 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False)['url']

source = get_stream_url(YOUTUBE_URL) if USE_YOUTUBE else 0
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(source)

video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
track_history = {}
completed_speeds = {}
seen_id_total = set()
frame_count = 0

def is_crossing_line(pos, line_pts):
    x, y = pos
    (x1, y1), (x2, y2) = line_pts
    if min(x1, x2) <= x <= max(x1, x2):
        line_y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        return abs(y - line_y) < 25
    return False

while cap.isOpened():
    for _ in range(SKIP_FRAMES):
        cap.grab()
        frame_count += 1

    ret, frame = cap.read()
    if not ret: break
    frame_count += 1

    # Фільтрація класів ТУТ (classes=...) прибирає будинки та зайві об'єкти
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        stream=True,
        classes=ALLOWED_CLASSES,
        conf=0.35
    )

    for r in results:
        if r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().numpy()
            clss = r.boxes.cls.int().cpu().numpy()

            for box, tid, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int(y2)
                class_name = model.names[cls]

                # Визначення швидкості
                if tid not in track_history and is_crossing_line((cx, cy), line1):
                    track_history[tid] = frame_count

                elif tid in track_history and tid not in completed_speeds and is_crossing_line((cx, cy), line2):
                    frames_passed = frame_count - track_history[tid]
                    if frames_passed > 0:
                        speed = (DISTANCE_METERS / (frames_passed / video_fps)) * 3.6 / 1.5
                        completed_speeds[tid] = round(speed, 1)
                        seen_id_total.add(tid)

                        with open(CSV_PATH, 'a', newline='') as f:
                            csv.writer(f).writerow([time.strftime("%H:%M:%S"), tid, class_name, completed_speeds[tid]])

                # ВІЗУАЛІЗАЦІЯ (Повернено ваш оригінальний стиль)
                color = (0, 255, 0) if tid not in completed_speeds else (255, 128, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Відображення тексту: швидкість або статус розрахунку
                speed_label = f"{completed_speeds[tid]} km/h" if tid in completed_speeds else "Calculating..."
                label = f"ID:{tid} {class_name} | {speed_label}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Загальний лічильник
    cv2.rectangle(frame, (20, 15), (320, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"TOTAL COUNT: {len(seen_id_total)}", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Лінії перетину
    cv2.line(frame, line1[0], line1[1], (0, 0, 255), 2)
    cv2.line(frame, line2[0], line2[1], (0, 0, 255), 2)

    display_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('Traffic AI Monitor', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()