import cv2
import numpy as np
from manager import WindowManager, CaptureManager
from tracker import Tracker
from detector import Detector

class Track(object):
    def __init__(self):
        self._windowManager = WindowManager('Track', self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)
        self._fgbg = cv2.BackgroundSubtractorMOG()
        self._objectFound = 0
        self._trackWindow = None
        self._tracker = Tracker()
        self._detector = Detector()

    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()

            # get a new frame 
            frame = self._captureManager.frame

            # detect objects by removing background
            found = self._detector.detect(frame)

            # updated tracked objects
            self._tracker.check(found) 

            # track objects
            tracked = self._tracker.track(frame)
            for t in tracked:
                t1x, t1y, t2x, t2y = t
                cv2.rectangle(frame, (t1x, t1y), (t2x, t2y), (0,0,255), 2)
                
            # track new objects if there is any
            for obj in found:
                p1x, p1y, p2x, p2y = obj
                if self._tracker.isTracked(obj):
                    cv2.rectangle(frame, (p1x, p1y), (p2x, p2y), (0,0,0), 2)
                    continue

                self._tracker.addroi(frame, obj) 
                cv2.rectangle(frame, (p1x, p1y), (p2x, p2y), (0,255,0), 2)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress (self, keycode):
        """Handle a keypress. 
        space  -> Take a screenshot.
        tab    -> Start/stop recording a screencast.
        escape -> Quit."""
        if keycode == 32: # space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9: # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27: # escape
            self._windowManager.destroyWindow()

if __name__=="__main__":
    Track().run()
