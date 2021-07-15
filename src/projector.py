import cv2
import numpy as np

class Projector:
    def __init__(self, point2=False):
        self.point1 = np.array([[0., 0.], [16.4, 13.9], [105, 68.], [52.5, 68], [88.6, 13.9]], dtype=np.float32) # field
        if not point2:
            self.point2 = np.array([[25.3, 59], [135.8, 77.2], [639.3, 19], [872.2, 688], [1141.5, 25.8]], dtype=np.float32) # frame
        else:
            self.point1 = np.array([[0., 0.], [16.4, 13.9], [52.5, 0.0], [88.6, 13.9], [105., 0.], [105, 68.], [52.5, 68], [0, 68]], dtype=np.float32)
            self.point2 = np.array([[0, 0],  [164, 152], [525., 0.0], [886., 152], [1050.0, 0.0], [1050, 699], [525, 699], [0, 699]], dtype=np.float32) # 2D-field
        self.H, _ = self.calculateHomoGraphy(make_rotate=point2)
        
    def calculateHomoGraphy(self, make_rotate = False):
        if not make_rotate:
            return cv2.findHomography(self.point2, self.point1, cv2.RANSAC)
        return cv2.findHomography(self.point1, self.point2, cv2.RANSAC)
    
    def project(self, p):
        point2 = np.dot(self.H, np.array([[p[0]], [p[1]], [1]], dtype=np.float32))
        point2 = (point2 / point2[-1, -1])
        return (point2[0, 0], point2[1, 0])
