import cv2
import numpy as np

portret = cv2.imread('photky/pic_fc.png')
portret = cv2.resize(portret, (portret.shape[1] // 2, portret.shape[0] // 2,))
portret = cv2.cvtColor(portret, cv2.COLOR_BGR2GRAY)
portret = cv2.Canny(portret, 150, 150)
# kernel = np.ones((5, 5), np.uint8)
# portret = cv2.dilate(portret, kernel, iterations = 1)
# portret = cv2.erode(portret, kernel, iterations = 1)

gmail = cv2.imread('photky/gmai.jpg')
gmail = cv2.resize(gmail, (gmail.shape[1] // 2, gmail.shape[0] // 2,))
gmail = cv2.cvtColor(gmail, cv2.COLOR_BGR2GRAY)
gmail = cv2.Canny(gmail, 350, 350)
kernel = np.ones((5, 5), np.uint8)
gmail = cv2.dilate(gmail, kernel, iterations = 1)
gmail = cv2.erode(gmail, kernel, iterations = 1)



cv2.imshow("portret", portret)
cv2.imshow("gmail", gmail)
cv2.waitKey(0)
cv2.destroyAllWindows()