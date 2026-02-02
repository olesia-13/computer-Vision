import cv2
import numpy as np
import os
import shutil


PROJECT_DIR = os.path.dirname(__file__)

IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')

OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
PEOPLE_DIR = os.path.join(OUTPUT_DIR, 'people')
NO_PEOPLE_DIR = os.path.join(OUTPUT_DIR, 'no_people')

os.makedirs(PEOPLE_DIR, exist_ok = True)
os.makedirs(NO_PEOPLE_DIR, exist_ok = True)

PROTOTXT_DIR = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt")
MODEL_DIR = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")


net = cv2.dnn.readNet(PROTOTXT_DIR, MODEL_DIR)



CLASSES = [ "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair","cow", "diningtable", "dog", "horse",
            "motorbike", "person", "pottedplant","sheep", "sofa", "train", "tvmonitor"]

PERSON_CLASS_ID = CLASSES.index("person")

CONF_THRESHOLD = 0.6

def detect_person(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor = 0.007843, size = (300, 300), mean = (127.5, 127.5, 127.5))
    net.setInput(blob)
    detections = net.forward()

    boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = detections[0, 0, i, 1]

        if class_id == PERSON_CLASS_ID and confidence >= CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7]

            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            boxes.append((x1, y1, x2, y2, confidence))

    return boxes


allowed_ext = (".jpg", ".jpeg", ".png", ".bmp")

files = os.listdir(IMAGES_DIR)

count_people = 0
count_no_people = 0
total_people_count = 0

for file in files:
    if not file.lower().endswith(allowed_ext):
        continue


    in_path = os.path.join(IMAGES_DIR, file)
    img = cv2.imread(in_path)

    boxes = detect_person(img)
    N = len(boxes)

    boxed = img.copy()

    for (x1, y1, x2, y2, conf) in boxes:
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(boxed,f"People count: {N}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2)

    if N > 0:
        count_people += 1
        total_people_count += N
        out_img = os.path.join(PEOPLE_DIR, "boxed_" + file)
    else:
        count_no_people += 1
        out_img = os.path.join(NO_PEOPLE_DIR, "boxed_" + file)

    cv2.imwrite(out_img, boxed)

print(f"People count: {total_people_count}")
