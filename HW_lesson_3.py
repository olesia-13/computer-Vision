import cv2
import numpy as np

portret = cv2.imread('images/pic_fc.png')
portret = cv2.resize(portret, (portret.shape[1] // 2, portret.shape[0] // 2,))
cv2.rectangle(portret, (60, 5), (330, 370), (84, 8, 211), 3)
cv2.putText(portret, "Osipova Olesia", (90, 420), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (84, 8, 211))

cv2.imshow("portret", portret)
cv2.waitKey(0)
cv2.destroyAllWindows()