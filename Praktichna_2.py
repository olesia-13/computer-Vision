import cv2
import numpy as np

img = cv2.imread("kr.jpg")
img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
img_copy = img.copy()
img = cv2.GaussianBlur(img, (3, 3), 5)

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_yel = np.array([18, 0, 215])
upper_yel = np.array([34, 255, 255])

lower_green = np.array([33, 52, 160])
upper_green = np.array([89, 255, 255])

lower_red = np.array([157, 59, 0])
upper_red = np.array([179, 255, 255])

lower_blue = np.array([78, 50, 0])
upper_blue = np.array([161, 255, 255])

mask_yel = cv2.inRange(img, lower_yel, upper_yel)
mask_blue = cv2.inRange(img, lower_blue, upper_blue)
mask_green = cv2.inRange(img, lower_green, upper_green)
mask_red = cv2.inRange(img, lower_red, upper_red)

mask_total = cv2.bitwise_or(mask_red, mask_blue, mask_yel)
mask_total = cv2.bitwise_or(mask_total, mask_green)



contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100:
        perimetr = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = round(w / h, 2)
        compactness = round((4 * np.pi * area) / (perimetr ** 2), 2)

        approx = cv2.approxPolyDP(cnt, 0.02 * perimetr, True)

        if len(approx) > 6 and compactness > 0.8:
            shape = "oval"
        else:
            shape = "inshe"

        cv2.drawContours(img_copy, [cnt], -1, (255, 255, 255), 2)
        cv2.circle(img_copy, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(img_copy, f'{shape}', [x, y - 20], cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 0, 0), 2)
        cv2.putText(img_copy, f'x: {int(cx)}, y: {int(cy)}', (x, y - 35), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 0, 0), 2)
        cv2.putText(img_copy, f'S: {int(area)}', (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 0, 0), 2)
        cv2.putText(img_copy, f'red', (285, 283), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 0, 0), 2)

        cv2.putText(img_copy, f'green', (330, 96), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 0, 0), 2)
        cv2.putText(img_copy, f'blue', (170, 167), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 0, 0), 2)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)



cv2.imshow("mask", img_copy)
cv2.imwrite("result.jpg", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()