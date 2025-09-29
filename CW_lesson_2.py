import cv2
import numpy as np

# image = cv2.imread('images/catto.jpg')
# # image = cv2.resize(image, (500, 250))
# image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
# print(image.shape)
# # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
# # image = cv2.flip(image, 1)
# # image = cv2.GaussianBlur(image, (9, 9), 7) # тільки непарні значення розмиття
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.Canny(image, 250, 250) # більші числа - менша чіткість
# # image = cv2.dilate(image, None, iterations = 1)
# kernel = np.ones((5, 5), np.uint8)
# print(image.shape)
# image = cv2.dilate(image, kernel, iterations = 1)
# image = cv2.erode(image, kernel, iterations = 1)
#
# cv2.imshow('catty', image)
# # cv2.imshow('catt', image[0:150, 0:100] )

# video = cv2.VideoCapture('video/doggy.mp4')
video = cv2.VideoCapture(0)

while True:
    success, frame = video.read()
    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# cv2.waitKey(0)
cv2.destroyAllWindows()
