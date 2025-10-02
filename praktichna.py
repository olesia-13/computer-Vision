import cv2
import numpy as np

img = np.zeros((400, 600, 3), np.uint8)
img[:] = 235, 239, 203

portret = cv2.imread('practic_photos/pic_fc.png')
portret = cv2.resize(portret, (120, 160))
img[25:25 + portret.shape[0], 25: 25 + portret.shape[1]] = portret


cv2.putText(img, "Olesia Osipova", (180, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
cv2.putText(img, "Computer Vision Student", (180, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
cv2.putText(img, "Email: osipolesia13@gmail.com", (180, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (73, 73, 155))
cv2.putText(img, "Phone: +380689182002", (180, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (73, 73, 155))
cv2.putText(img, "30/04/2010", (180, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (73, 73, 155))
cv2.putText(img, "OpenCV Business Card", (140, 370), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

qr = cv2.imread('practic_photos/qr_code.png')
qr = cv2.resize(qr, (100, 100))
img[230:230 + qr.shape[0], 482: 482 + qr.shape[1]] = qr

cv2.rectangle(img, (10, 10), (590, 390), (156, 42, 49), 2)



cv2.imshow("visitivka", img)
cv2.imwrite("business_card.png", img)
cv2.waitKey(0)
cv2.destroyAllWindows()