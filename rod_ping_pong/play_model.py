import cv2
from ultralytics import YOLO
import ultralytics
import time

from annotation.utils import generate_annotations, generate_images, load_classes

def draw_ball(input_frame, ult_result: ultralytics.engine.results.Results, ball_class_index = 0):
    frame = input_frame.copy()
    cls = ult_result.boxes.cls.cpu().numpy().astype(int)
    conf = ult_result.boxes.conf.cpu().numpy().astype(float)
    boxes = ult_result.boxes
    ball_conf = [conf[i] for i in range(len(cls)) if cls[i] == ball_class_index]
    ball_boxes = [boxes[i] for i in range(len(cls)) if cls[i] == ball_class_index]
    highest_conf_index = ball_conf.index(max(ball_conf))
    highest_conf_ball_box = ball_boxes[highest_conf_index]
    x, y, w, h = highest_conf_ball_box.xywh.cpu().numpy().astype(float)[0].tolist()
    x_center = x + w / 2
    y_center = y + h / 2
    # draw ball using cv2 and centers
    cv2.circle(input_frame, (int(x_center), int(y_center)), 10, (0, 255, 0), 2)

    return input_frame


    frame = input_frame.copy()
    for result in ult_result:
        if result["name"] == ball_class_name:
            x_center = int(result["relative_coordinates"]["center_x"] * frame.shape[1])
            y_center = int(result["relative_coordinates"]["center_y"] * frame.shape[0])
            width = int(result["relative_coordinates"]["width"] * frame.shape[1])
            height = int(result["relative_coordinates"]["height"] * frame.shape[0])
            cv2.rectangle(frame, (x_center - width // 2, y_center - height // 2), (x_center + width // 2, y_center + height // 2), (0, 255, 0), 2)
            cv2.putText(frame, ball_class_name, (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


def draw_annotation(input_frame, annotations, classes,):
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
        
        #if classes[class_index] == "player":
        #    if first_player:
        #        first_player = False
        #        first_player_width = width - width % 2
        #        frame[0:height - height % 2, 0:width - width % 2, 0:3] = \
        #            original_frame[(y_center - height // 2):(y_center + height // 2), (x_center - width // 2):(x_center + width // 2), 0:3]
        #    else:
        #        frame[0:height - height % 2, first_player_width:first_player_width+ width - width % 2, 0:3] = \
        #            original_frame[(y_center - height // 2):(y_center + height // 2), (x_center - width // 2):(x_center + width // 2), 0:3]
    return frame



def main():
    # Load the YOLOv8 model
    #MODEL_WEIGHTS = 'rod_ping_pong/training/saved/2hrs_30fps_1to2min_0.83x_speed_v8n_20epochs/weights/best.pt'
    MODEL_WEIGHTS = 'rod_ping_pong/training/saved/2hrs_30fps_1to2min_0.83x_speed_v8n_10epochs/weights/best.pt'
    # MODEL_WEIGHTS = 'best-colab-10epochs.pt'
    # MODEL_WEIGHTS = 'yolov8n-pose.pt'
    model = YOLO(MODEL_WEIGHTS, task="detect")

    ON_ANNOTATED_IMAGES = False
    if ON_ANNOTATED_IMAGES:
        YOLO_DATA_DIR = r"C:\Users\yannick.gibson\projects\school\BP\bachelors_thesis\annotation\mydata\yolo_output\blurry\2hrs_30fps_1to2min_0.83x_speed" 
        annotations_dir = YOLO_DATA_DIR + r"\labels"
        images_dir = YOLO_DATA_DIR + r"\images"
        classes_dir = YOLO_DATA_DIR + r"\classes.txt"

        images_generator = generate_images(images_dir)
        annotations_generator = generate_annotations(annotations_dir)
        classes = load_classes(classes_dir)
        for frame, annotation in zip(images_generator, annotations_generator):


            results = model.predict(frame, verbose=False)  # define what classes to predict
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
        #VIDEO_PATH = r"C:\Users\yannick.gibson\projects\work\important\ball-tracker\data\videos\blurry\2hrs.ts"
        VIDEO_PATH = r"C:\Users\yannick.gibson\projects\work\important\ball-tracker\data\videos\blurry\42min.mp4"
        #VIDEO_PATH = "C:/Users/yannick.gibson/projects/work/important/ball-tracker/data/videos/old/ping_03_cam_2.mp4"
        #VIDEO_PATH = "C:/Users/yannick.gibson/projects/work/important/ball-tracker/data/videos/old/ping_05_cam_2.mp4"
        SHOW_EVERY = 3
        # Open the video file
        cap = cv2.VideoCapture(VIDEO_PATH)
        i = 0


        while cap.isOpened():


            img_start_time = time.time()
            img_time = time.time() - img_start_time
            
            #cap.set(cv2.CAP_PROP_POS_FRAMES, i * SHOW_EVERY)

            # Read a frame from the video
            success, frame = cap.read()
            i += 1
            if i < 100:
                print("skipping: ", i)
                continue

            if success:

                if i % SHOW_EVERY == 0:
                    #frame = frame[250:-650, 500:-600, :]
                    results = model.predict(frame, verbose=False, classes=[0], conf=0.4)  # define what classes to predict
                    annotated_frame = results[0].plot(img=frame)  # img = zeros

                    annotated_frame = draw_ball(annotated_frame, results[0])

                    cv2.imshow("frame", annotated_frame)

                    img_time = time.time() - img_start_time
                    img_start_time = time.time()
                    print(f"({i}th frame)fps: {1/img_time:.3f}")
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                    if key == ord('p'):
                        while True:
                            if cv2.waitKey(1) & 0xFF == ord('p'):
                                break
                    # if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'):

            

        else:
            print("Could not open video")


if __name__ == "__main__":
    main()