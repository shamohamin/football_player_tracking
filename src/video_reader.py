from time import time
import cv2
import os

CROP_AREA = 90

def preprocess(frame):
    frame = frame[CROP_AREA:]
    frame[:27, :frame.shape[1] // 3] = 0
    frame[25:35,:int(frame.shape[1] // 3.5)] = 0
    frame[35:45,:int(frame.shape[1] // 5.5)] = 0
    frame[45:46,:frame.shape[1] // 6] = 0
    
    return frame

def reading_frames():
    video_pathname = os.path.join(os.getcwd(), "assets", "output.mp4")
    video_cap  = cv2.VideoCapture(video_pathname)
    width  = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # video_cap.set(cv2.CAP_PROP_FPS, 15)
    # fps = video_cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break
        
        frame = preprocess(frame)
        frames.append(frame)
        
    if len(frames) == 0:
        raise Exception("somthing went wrong for reading the frames of image.!")

    video_cap.release()
    
    return frames, width, height
    
    
def showing_video(frames: list, delay: int):
    for frame in frames:
        cv2.imshow("video", frame)
        key = cv2.waitKey(delay)
        if key == ord("q") & 0xFF:
            break
        
