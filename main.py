from src.video_reader import CROP_AREA
from src.video_reader import reading_frames
from src.projector import Projector
from src.backgroundSubtractor import BackgroundSubtractor
from src.model import Model
import os
import cv2
import numpy as np
import time

BLUE, WHITE, REF = 0, 1, 2

def reading_asset(filepath, gray_scale=False):
    img_pathname = os.path.join(os.getcwd(), "assets", filepath)
    field_photo = cv2.imread(img_pathname, cv2.IMREAD_GRAYSCALE if gray_scale else cv2.IMREAD_UNCHANGED)
    return field_photo

def get_blobs(thresh, iterations=1):
    k1 = np.array([ [1,1,1,1],
                    [1,1,1,1],
                    [1,1,1,1],
                    [0,1,1,0],
                    [0,1,1,0],
                    [1,1,1,1],
                    [1,1,1,1],
                    [1,1,1,1]], dtype=np.uint8)

    k2 = np.ones((25, 15))
    up_area = thresh[:int(thresh.shape[0] / 8)]

    up_area = cv2.erode(up_area, k1)
    up_area = cv2.dilate(up_area, k2)

    k1 = np.array([ [1,1,1,1],
                    [1,1,1,1],
                    [0,1,1,0],
                    [0,1,1,0],
                    [0,1,1,0],
                    [1,1,1,1]], dtype=np.uint8)
    k2 = np.ones((30, 20), dtype=np.uint8)
    
    middel_area = thresh[int(thresh.shape[0] / 8):int(thresh.shape[0] / 3.5)]
    middel_area = cv2.erode(middel_area, k1)
    middel_area = cv2.dilate(middel_area, k2)
    
    down_area = thresh[int(thresh.shape[0] / 3.5):]
    k2 = np.ones((60, 30), np.uint8)
    down_area = cv2.morphologyEx(down_area, cv2.MORPH_CLOSE, k2, iterations=iterations)

    thresh = np.vstack([up_area, middel_area, down_area])
    # Calculate the centers of the contours.
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

    return thresh, contours


def main2():
    frames, width, height = reading_frames()
    # exit(2)
    bg_sub = BackgroundSubtractor()
    proj_field_to_top = Projector()
    proj_field_2d     = Projector(point2=True)
    D_field_photo = reading_asset("2D_field.png")
    roi_count = 0
    model = Model()
    
    
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0
    
    for frame in frames:
        frame_cop = frame.copy()
        new_frame_time = time.time()
        
        frame = cv2.GaussianBlur(frame, (13, 13), 0)
        J = D_field_photo.copy()
        
        mask = bg_sub.applySubtractor(frame)
        
        mask, contours = get_blobs(mask, 1)
        
        pp = {
            "rois": [],
            "points": []
        }
        
        for c in contours: 
            rect, area = cv2.boundingRect(c), cv2.contourArea(c)
            x,y,w,h = rect
            
            if (y >= (height // 3.5) and h <= 50):
                continue
            if ((height // 3.5) >= y >= int(height / 8) and h <= 35):
                continue
            
            cv2.putText(frame_cop, f"{area}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
            cv2.putText(frame_cop, f"{h} {w}", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            try:
                roi = frame[y:y+h, x:x+w].copy()
                roi = np.asarray(roi, np.float32)
                roi = cv2.resize(roi, (80, 80))
                pp["rois"].append(roi)                
                pp["points"].append(proj_field_to_top.project((x+w, y+h)))
            except:
                pass
            roi_count += 1
            cv2.rectangle(frame_cop, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        labels = []
        if len(pp["rois"]) >= 1:
            rois = np.array(pp["rois"])
            labels = model.predict(rois)
        
        
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame_cop, str(int(fps)), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, [0,255,0], 2)
        if len(pp["points"]) >= 1:
            
            for i, p in enumerate(pp["points"]):
                x, y = proj_field_2d.project((p[0], p[-1]))

                if labels[i] == BLUE:
                    cv2.circle(J, (int(x), int(y)), 7, (255, 0, 0), -1)
                elif labels[i] == WHITE:
                    cv2.circle(J, (int(x), int(y)), 7, (0, 0, 255), -1)
                elif labels[i] == REF:
                    cv2.circle(J, (int(x), int(y)), 7, (0, 255, 255), -1)
        
       
        cv2.imshow("frame", frame_cop)
        cv2.imshow("2d", J)
        key = cv2.waitKey(0) 
        
        if key & 0xFF == ord('q'):
            break
    
if __name__ == '__main__':
    main2()
    
