import cv2

class BackgroundSubtractor:
    def __init__(self):
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    
    def applySubtractor(self, frame):
        return self.fgbg.apply(frame)