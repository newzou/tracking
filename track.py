import cv2
import numpy as np
from manager import WindowManager, CaptureManager

class Track(object):
    def __init__(self):
        self._windowManager = WindowManager('Track', self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)
        self._fgbg = cv2.BackgroundSubtractorMOG()

    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            original = np.copy(frame)
            fgmask = self._fgbg.apply(frame)
            ret, thresh = cv2.threshold(fgmask,60 ,255,0)
            contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   
            for c in contours:
                if cv2.contourArea(c) > 1000:
                    cv2.drawContours(frame, c, -1, (0,255,0), 3)

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
