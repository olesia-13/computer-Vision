import cv2
import os
from collections import Counter
import pandas as pd

#1 завантажуємо модель
net = cv2.dnn.readNetFromCaffe("data/MobileNet/mobilenet_deploy.prototxt",
                               "data/MobileNet/mobilenet.caffemodel")

#2 зчитуємо список назв класів
classes = []
with open("data/MobileNet/synset.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)
# 3. Папка з фото
image_dir = "images/MobileNet"

# 4. Список файлів
image_files = [f for f in os.listdir(image_dir)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]

results = []

# 5. Обробка кожного фото
for filename in image_files:
    image_path = os.path.join(image_dir, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Не вдалося відкрити {filename}")
        continue

    # 6. Підготовка зображення
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (224, 224)),
        1.0 / 127.5,
        (224, 224),
        (127.5, 127.5, 127.5)
    )

    # 7. Класифікація
    net.setInput(blob)
    preds = net.forward()

    # 8. Визначаємо клас
    idx = preds[0].argmax()
    label = classes[idx] if idx < len(classes) else "unknown"
    conf = float(preds[0][idx]) * 100

    # 9. Виводимо у консоль
    print(f"{filename}: {label} ({conf:.2f}%)")

    # 10. Підписуємо фото
    text = f'{label}: {int(conf)}%'
    cv2.putText(image, text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow(filename, image)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    results.append(label)

# 11. Таблиця результатів
counts = Counter(results)
df = pd.DataFrame(list(counts.items()), columns=["Клас", "Кількість"])
print("\nТаблиця результатів:")
print(df)