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
