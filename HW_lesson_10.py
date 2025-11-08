import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import deque

# 2. Формуємо набори даних ------------------------------------------------------------
X = []  # X - список ознак
y = []  # y - список міток - правильні відповіді


colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "da-blue": (255, 0, 0),
    "yellow": (0, 208, 255),
    "li-blue": (255, 255, 0),
    "white": (255, 255, 255),
    "orange":(255, 132, 0),
    "pink": (199, 199, 255)
}

# Створюємо шумові зразки для навчання -----------------------------------------------
for color_name, bgr in colors.items():
    for i in range(20):
        noise = np.random.randint(-20, 20, 3)
        sample = np.clip(np.array(bgr) + noise, 0, 255)
        X.append(sample)
        y.append(color_name)

# 3. Розділяємо дані за пропорцією 70:30 ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# 4. Навчаємо модель -----------------------------------------------------------------
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)


smooth_buffer = deque(maxlen=5)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Маска для відокремлення кольору
    mask = cv2.inRange(hsv, (20, 50, 50), (255, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[y:y + h, x:x + w]

            # Середнє значення кольору в області
            mean_color = cv2.mean(roi)[:3]
            smooth_buffer.append(mean_color)

            # Згладження: середнє за кілька останніх кадрів
            smoothed_color = np.mean(smooth_buffer, axis=0).reshape((1, -1))

            # Отримуємо ймовірності
            probs = model.predict_proba(smoothed_color)[0]
            label_index = np.argmax(probs)
            label = model.classes_[label_index]
            confidence = probs[label_index]


            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            text = f"{label.upper()} ({confidence*100:.1f}%)"
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("color", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
