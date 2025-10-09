import cv2
import numpy as np

img = cv2.imread("images/ginger.jpg")
img_copy = img.copy()
img = cv2.GaussianBlur(img, (3, 3), 5)

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([0, 57, 97]) # мінімальний поріг
upper = np.array([86, 255, 255]) # максимальний поріг
mask = cv2.inRange(img, lower, upper)
img = cv2.bitwise_and(img, img, mask = mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 150:
        perimetr = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)

        if M["m00"] != 0:   # центр мас
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])


        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = round(w / h, 2) # допомагає відрізняти співвідношення сторін
        compactness = round((4 * np.pi * area) / (perimetr ** 2), 2) # міра округлості об'єкта

        approx = cv2.approxPolyDP(cnt, 0.02 * perimetr, True) # чим більше значення 0.02, тим більще вершин. чим менше - тим краще
        if len(approx) == 3:
            shape = "trikutnik"
        elif len(approx) == 4:
            shape = "square"
        elif len(approx) > 8:
            shape = "oval"
        else:
            shape = "inshe"

        cv2.drawContours(img_copy, [cnt], -1, (255, 255, 255), 2)
        cv2.circle(img_copy, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(img_copy, f'{shape}', [x, y - 20], cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 0, 0), 2)
        cv2.putText(img_copy, f'A: {int(area)}, P: {int(perimetr)}', (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 0, 0), 2)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img_copy, f'AR: {aspect_ratio}, C: {compactness}', (x, y - 35), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 0, 0), 2)


cv2.imshow("mask", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()