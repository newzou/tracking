import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fgbg = cv2.BackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()

    if ret == True:
        original = np.copy(frame)

        fgmask = fgbg.apply(frame)

        ret, thresh = cv2.threshold(fgmask,60 ,255,0)

        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   
        for c in contours:
            if cv2.contourArea(c) > 1000:
                cv2.drawContours(original, c, -1, (0,255,0), 3)

        cv2.imshow('image', original)
        
    k = cv2.waitKey(30) & 0xff
    # ESC to close
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
