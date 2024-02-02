import cv2
from ultralytics import YOLO
#from project_name.find_table.locating_table import get_table_corners
import numpy as np
import time

# Load the YOLOv8 model
#model = YOLO('runs/detect/train2/weights/best.pt')
#model = YOLO('best-colab-10epochs.pt')
#"yolov8n-pose.pt"
model = YOLO('training/saved/best775-early-stopping.pt')
# model = YOLO('yolov8n-pose.pt')
# model = YOLO('models/yolov5s.pt')

# Open the video file
print(__file__)
video_path = "C:/Users/yannick.gibson/projects/work/important/ball-tracker/videos/ping_03_cam_2.mp4"
#video_path = "C:/Users/yannick.gibson/projects/work/important/ball-tracker/videos/ping_05_cam_2.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames


while cap.isOpened():

    img_start_time = time.time()

    img_time = time.time() - img_start_time

    # Read a frame from the video
    success, frame = cap.read()

    if success:
        results = model(frame, verbose=False)  # define what classes to predict
        annotated_frame = results[0].plot(img=frame) # img = zeros

        cv2.imshow("frame", annotated_frame)


        img_time = time.time() - img_start_time
        img_start_time = time.time()
        print(f"fps: {1/img_time:.3f}")
        if cv2.waitKey(1) == ord('q'):
            break
        # if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'):
else:
    print("Could not open video")

