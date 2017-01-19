import cv2
import numpy as np
import time

class Detector(object):
    def __init__(self, background=None):
        self._background = background
        self._fgbg = cv2.BackgroundSubtractorMOG()

    @property
    def background(self):
        return self._background

    @background.setter    
    def widnow(self, bg):
        self._background = bg

    def setBackground(self, bg):
        self._background = bg

    def hasbg(self):
        return self._background is not None

    def detect(self, frame):
        fgmask = self._fgbg.apply(frame)
        ret, thresh = cv2.threshold(fgmask, 200, 255, 0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
       
        found = []
        for c in contours:
            if cv2.contourArea(c) > 2000:
                 x,y,w,h = cv2.boundingRect(c)
                 found.append(((x,y),(x+w,y+h)))

        return found

            #if self._tracker.isTracked((cx, cy, cx+cw, cy+ch)):
            #    continue


#            count += 1
#            if p1x is None or cx < p1x:
#                p1x = cx
#            if p1y is None or cy < p1y:
#                p1y = cy
#            if p2x is None or p2x < cx+cw:
#                p2x = cx+cw
#            if p2y is None or p2y < cy+ch:
#                p2y = cy+ch
#
#        if self._window is None:
#            return False
#        x1, y1, x2, y2 = rect
#        wx1, wy1, wx2, wy2 = self._window
#        if x2 < wx1 or wx2 < x1:
#            return False 
#        if y2 < wy1 or wy2 < y1:
#            return False 
#        return True
       
