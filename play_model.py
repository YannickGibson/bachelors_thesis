import cv2
from ultralytics import YOLO
import numpy as np
import time
from os import listdir
from os.path import isfile, join
from typing import Iterator

def generate_annotations(annotations_path: str) -> Iterator[list]:

    # "listdir" provides sorted file names
    onlyfiles = [f for f in listdir(annotations_path) if isfile(join(annotations_path, f))]
    for file in onlyfiles:
        with open(annotations_path + "\\" + file, 'r') as f:
            annotation = f.read().split("\n\n")

        yield annotation[:-1] # ommit last one, because it is an empty string

def generate_images(images_path: str) -> Iterator[np.array]:
    # "listdir" provides sorted file names
    onlyfiles = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    for file in onlyfiles:
        img = cv2.imread(images_path + "\\" + file, cv2.IMREAD_COLOR)
        yield img

def load_classes(classes_path: str) -> list:
    with open(classes_path, 'r') as f:
        classes = f.read().split("\n")
    return classes

def draw_annotation(input_frame, annotations, classes, original_frame):
    frame = input_frame.copy()
    first_player = True
    first_player_width = None
    for annotation in annotations:
        # Split the annotation into its components
        annotation = annotation.split(" ")
        class_index = int(annotation[0])
        x_center = int(float(annotation[1]) * frame.shape[1])
        y_center = int(float(annotation[2]) * frame.shape[0])
        width = int(float(annotation[3]) * frame.shape[1])
        height = int(float(annotation[4]) * frame.shape[0])

        # Draw the bounding box
        cv2.rectangle(frame, (x_center - width // 2, y_center - height // 2), (x_center + width // 2, y_center + height // 2), (0, 255, 0), 2)

        # Draw the class name
        cv2.putText(frame, classes[class_index], (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if classes[class_index] == "player":
            if first_player:
                first_player = False
                first_player_width = width - width % 2
                frame[0:height - height % 2, 0:width - width % 2, 0:3] = \
                    original_frame[(y_center - height // 2):(y_center + height // 2), (x_center - width // 2):(x_center + width // 2), 0:3]
            else:
                frame[0:height - height % 2, first_player_width:first_player_width+ width - width % 2, 0:3] = \
                    original_frame[(y_center - height // 2):(y_center + height // 2), (x_center - width // 2):(x_center + width // 2), 0:3]
    return frame



def main():
    # Load the YOLOv8 model
    #MODEL_WEIGHTS = 'training/saved/2hrs_30fps_1to2min_0.83x_speed_v8n_20epochs/weights/best.pt'
    MODEL_WEIGHTS = 'training/saved/2hrs_30fps_1to2min_0.83x_speed_v8n_10epochs/weights/best.pt'
    # MODEL_WEIGHTS = 'best-colab-10epochs.pt'
    # MODEL_WEIGHTS = 'yolov8n-pose.pt'
    model = YOLO(MODEL_WEIGHTS)

    ON_ANNOTATED_IMAGES = True
    if ON_ANNOTATED_IMAGES:
        YOLO_DATA_DIR = r"C:\Users\yannick.gibson\projects\school\BP\bachelors_thesis\annotation\mydata\ls2yolo_output\2hrs_30fps_1to2min_0.83x_speed" 
        annotations_dir = YOLO_DATA_DIR + r"\labels"
        images_dir = YOLO_DATA_DIR + r"\images"
        classes_dir = YOLO_DATA_DIR + r"\classes.txt"

        images_generator = generate_images(images_dir)
        annotations_generator = generate_annotations(annotations_dir)
        classes = load_classes(classes_dir)
        for frame, annotation in zip(images_generator, annotations_generator):


            results = model(frame, verbose=False)  # define what classes to predict
            predicted_frame = results[0].plot(img=frame)  # img = zeros
            #predicted_frame = frame

            annotated_predicted_frame = draw_annotation(predicted_frame, annotation, classes, original_frame=frame)
            #annotated_predicted_frame = predicted_frame

            cv2.imshow("frame", annotated_predicted_frame)
            
            #if cv2.waitKey(1000 // 50) == ord('q'):
            #k = cv2.waitKey(30) & 0xff
            if cv2.waitKey(30) == ord('q'):
                break
    else:
        #VIDEO_PATH = r"C:\Users\yannick.gibson\projects\work\important\ball-tracker\data\videos\blurred\2hrs.ts"
        VIDEO_PATH = r"C:\Users\yannick.gibson\projects\work\important\ball-tracker\data\videos\blurry\42min.ts"
        #VIDEO_PATH = "C:/Users/yannick.gibson/projects/work/important/ball-tracker/data/videos/ping_03_cam_2.mp4"
        #VIDEO_PATH = "C:/Users/yannick.gibson/projects/work/important/ball-tracker/data/videos/ping_05_cam_2.mp4"
        SHOW_EVERY = 40
        # Open the video file
        cap = cv2.VideoCapture(VIDEO_PATH)
        i = 0

        # video capture playback in reverse
        #cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)

        while cap.isOpened():

            img_start_time = time.time()

            img_time = time.time() - img_start_time

            
            #cap.set(cv2.CAP_PROP_POS_FRAMES, i * SHOW_EVERY)

            # Read a frame from the video
            success, frame = cap.read()
            if success:

                if i % SHOW_EVERY == 0:
                    results = model(frame, verbose=False)  # define what classes to predict
                    annotated_frame = results[0].plot(img=frame)  # img = zeros

                    cv2.imshow("frame", annotated_frame)

                    img_time = time.time() - img_start_time
                    img_start_time = time.time()
                    print(f"fps: {1/img_time:.3f}")
                    if cv2.waitKey(1) == ord('q'):
                        break
                    # if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'):

                i += 1
            

        else:
            print("Could not open video")


if __name__ == "__main__":
    main()