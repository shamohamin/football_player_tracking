from src.video_reader import CROP_AREA
from src.video_reader import reading_frames
from src.projector import Projector
from src.backgroundSubtractor import BackgroundSubtractor
from src.model import Model
import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt

BLUE, WHITE, REF = 0, 1, 2


def track(bboxes, frame):
    multiTracker = cv2.legacy.MultiTracker_create()

    for bbox in bboxes:
        multiTracker.add(cv2.legacy.TrackerCSRT_create(), frame, bbox)

    return multiTracker

def reading_asset(filepath, gray_scale=False):
    img_pathname = os.path.join(os.getcwd(), "assets", filepath)
    field_photo = cv2.imread(img_pathname, cv2.IMREAD_GRAYSCALE if gray_scale else cv2.IMREAD_UNCHANGED)
    return field_photo

def get_blobs(thresh, iterations=1):
    # kernel = np.ones((33, 29), np.uint8)

    k1 = np.array([ [0,1,1,0],
                    [0,1,1,0],
                    [0,1,1,0],
                    [0,1,1,0],
                    [0,1,1,0],
                    [0,1,1,0],
                    [0,1,1,0],
                    [0,1,1,0]], dtype=np.uint8)

    k2 = np.ones((25, 15))
    # k2 = np.ones((40, 25))
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
    # up_area = cv2.morphologyEx(up_area, cv2.MORPH_CLOSE, k1, iterations=iterations)
    
    # kernel = np.ones((40, 40), np.uint8)
    # kernel[kernel.shape[0] // 4:kernel.shape[0] - kernel.shape[0] // 4, (0, -1)] = 0
    down_area = thresh[int(thresh.shape[0] / 3.5):]
    k2 = np.ones((60, 30), np.uint8)
    down_area = cv2.morphologyEx(down_area, cv2.MORPH_CLOSE, k2, iterations=iterations)
    # down_area = cv2.erode(down_area, k1)
    # down_area = cv2.dilate(down_area, k2)
    
    thresh = np.vstack([up_area, middel_area, down_area])
    # cv2.imshow("myThresh", thresh)
    # Calculate the centers of the contours.
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

    return thresh, contours



def main2():
    frames, width, height = reading_frames()
    bg_sub = BackgroundSubtractor()
    proj_field_to_top = Projector()
    proj_field_2d     = Projector(point2=True)
    D_field_photo = reading_asset("2D_field.png")
    roi_count = 0
    # model = Model()
    
    count = 0
    tracker = None
    
    for frame in frames:
        frame_cop = frame.copy()
        J = D_field_photo.copy()
        
        if count % 30 == 0:
            print("id")
            frame = cv2.GaussianBlur(frame, (13, 13), 0)
            
            
            mask = bg_sub.applySubtractor(frame)
            # cv2.imshow("m", mask)

            # _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            mask, contours = get_blobs(mask, 1)
            # points, rois = [], []
            pp = {
                "rois": [],
                "points": []
            }
            # counter = 0
            detections = []
            for c in contours: 
                rect, area = cv2.boundingRect(c), cv2.contourArea(c)
                x,y,w,h = rect
                
                

                if (y >= (height // 3.5) and h <= 50):
                    continue
                if ((height // 3.5) >= y >= int(height / 8) and h <= 35):
                    continue
               
                detections.append((x, y, w, h))
                try:
                    roi = frame[y:y+h, x:x+w].copy()
                    roi = cv2.resize(roi, (80, 80))
                    # roi = roi.astype("float32") / 255.
                    pp["rois"].append(roi)
                    # print(cv2.imwrite(f"./roi_2/{roi_count}.jpg", roi))
                    # counter += 1
                    pp["points"].append(proj_field_to_top.project((x+w, y+h)))
                    # cv2.imshow("roi", roi)
                    # cv2.waitKey()
                    # rois.append(roi)
                    # points.append(proj_field_to_top.project((x+w+CROP_AREA, y+h+CROP_AREA)))
                except:
                    pass
                roi_count += 1

            

                # print("-------------------------------")
                # print(detections)
                # print("-------------------------------")

                tracker = track(detections, frame)
            
        if tracker != None:
            ok, boxes = tracker.update(frame)
            
            print(len(boxes))
            for i, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                # cv2.rectangle(frame_cop, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.rectangle(frame_cop, p1, p2, (0, 255, 0), 2)

                # cv2.rectangle(frame_cop, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # labels = []
        # if len(pp["rois"]) >= 1:
        #     rois = np.array(pp["rois"])
        #     labels = model.predict(rois)
            # print(labels)
        
        if len(pp["points"]) != 0:
            # print(len(pp["points"]))
            # print(len(pp["rois"]))
            for i, p in enumerate(pp["points"]):
                x, y = proj_field_2d.project((p[0], p[-1]))
                cv2.circle(J, (int(x), int(y)), 7, (255, 0, 0), -1)

        #         if labels[i] == BLUE:
        #             cv2.circle(J, (int(x), int(y)), 7, (255, 0, 0), -1)
        #         elif labels[i] == WHITE:
        #             cv2.circle(J, (int(x), int(y)), 7, (0, 0, 255), -1)
        #         elif labels[i] == REF:
        #             cv2.circle(J, (int(x), int(y)), 7, (0, 255, 255), -1)
            
       
        cv2.imshow("frame", frame_cop)
        cv2.imshow("2d", J)
        # cv2.imshow("mask", mask)
        key = cv2.waitKey(1000//33) # 30 frame
        count += 1
        if key == ord('q'):
            break
    
if __name__ == '__main__':
    main2()
    # model = Model()
    # I = cv2.imread("./my_test/blue_2.png")
    # I = cv2.resize(I, (80, 80))
    # I = I.astype("float32")/255
    
    # print(model.predict(np.reshape(I, (1, *I.shape))))
    