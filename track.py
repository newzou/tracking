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
            frame = self._captureManager.frame

            #if self._detector.background():
            if not self._detector.hasbg():
                self._detector.setBackground(frame)
            else:
                found = self._detector.detect(frame)
                for obj in found:
                    p1, p2 = obj
                    cv2.rectangle(frame, p1, p2, (255,0,0), 2)

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
