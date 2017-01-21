import cv2
import numpy as np
import time

class Detector(object):
    SIMPLE = 1

    def __init__(self, background=None, method = SIMPLE):
        self._background = background
        self._fgbg = cv2.BackgroundSubtractorMOG()
        self._method = method

    @property
    def background(self):
        return self._background

    @background.setter    
    def widnow(self, bg):
        self._background = bg

    def updateBackground(self, bg):
	gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
        self._background = gray 

    def detect(self, frame):
        """return rectangle contours of frame by removing background"""
        if self._method != Detector.SIMPLE:
            pass
        else:
            return self.detect_simple(frame)

    def detect_simple(self, frame):
        
        if self._background is None:
            self.updateBackground(frame)
            return []

        # preprocess frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # compute the difference between current frame and background
        delta = cv2.absdiff(self._background, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=2)
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE)
     
        found = []
        for c in cnts:
            if cv2.contourArea(c) > 900:
                 x,y,w,h = cv2.boundingRect(c)
                 found.append((x,y,x+w,y+h))

        return found

    
