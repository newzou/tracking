import cv2
import numpy as np
import time

class Roi(object):
    def __init__(self, roi, color=(255, 0, 0), window=None):
        self._roi = roi
        self._color = color
        self._window = window

    @property
    def roi(self):
        return self._roi

    @property
    def color(self):
        return self._color
  
    @property
    def window(self):
        return self._window

    @window.setter    
    def widnow(self, win):
        self._window = win

    def setWindow(self, win):
        self._window = win

    def overlap(self, rect):
        if self._window is None:
            return False
        x1, y1, x2, y2 = rect
        wx1, wy1, wx2, wy2 = self._window
        if x2 < wx1 or wx2 < x1:
            return False 
        if y2 < wy1 or wy2 < y1:
            return False 
        return True
       
        
class Tracker(object):
    def __init__(self):
        self._rois = []        
        self._termcrit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    @property
    def rois(self):
        return self._rois

    def addroi(self, frame, rect):
        x1, y1, x2, y2 = rect
        hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_roi, np.array((100., 30., 32.)), np.array((180., 120., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        self._rois.append(Roi(roi_hist, window=rect))

    def track(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if len(self._rois) > 0:
            dst = cv2.calcBackProject([hsv], [0], self._rois[0].roi, [0,180], 1)
            
            # apply meanshift to get the new location
            x1, y1, x2, y2 = self._rois[0].window
            track_window = (x1, y1, x2-x1, y2-y1)
            ret, track_window = cv2.meanShift(dst, track_window, self._termcrit)
            x,y,w,h = track_window

            self._rois[0].setWindow((x,y,x+w,y+h))

    def isTracked(self, rect):
        for roi in self.rois:        
            if roi.overlap(rect):
                return True
        return False