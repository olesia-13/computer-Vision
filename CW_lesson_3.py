import cv2
import numpy as np

img = np.zeros((512, 400, 3), np.uint8)
#rgb = bgr
# img[:] = 142, 228, 126 # все залити
# img[100:150, 200:280] = 142, 228, 126 # залити фрагмент 100 - y, 150 - x

cv2.rectangle(img, (100, 100), (200, 200), (142, 228, 126), 1)
cv2.line(img, (100, 100), (200, 200), (142, 228, 126), 1)
print(img.shape)
cv2.line(img, (0, img.shape[0] // 2), (img.shape[1], img.shape[0] // 2), (142, 228, 126), 1)
cv2.line(img, (img.shape[1] // 2, 0), (img.shape[1] // 2, img.shape[0]), (142, 228, 126), 1)



cv2.circle(img, (200, 200), 20, (142, 228, 126), 1)

cv2.putText(img, "Komarov Ivan", (200, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (142, 228, 126))

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()