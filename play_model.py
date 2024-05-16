import cv2
from ultralytics import YOLO
import ultralytics
import numpy as np
import os
from tqdm import tqdm

from utils import generate_annotations, generate_images, load_classes, display_frame, draw_annotations


def get_ball_xy(ult_result: ultralytics.engine.results.Results, ball_class_index: int = 0):
    cls = ult_result.boxes.cls.cpu().numpy().astype(int)
    conf = ult_result.boxes.conf.cpu().numpy().astype(float)
    boxes = ult_result.boxes
    ball_conf = [conf[i] for i in range(len(cls)) if cls[i] == ball_class_index]
    if len(ball_conf) == 0:
        return None, None
    ball_boxes = [boxes[i] for i in range(len(cls)) if cls[i] == ball_class_index]
    highest_conf_index = ball_conf.index(max(ball_conf))
    highest_conf_ball_box = ball_boxes[highest_conf_index]
    x_center, y_center, w, h = highest_conf_ball_box.xywh.cpu().numpy().astype(float)[0].tolist()
    return x_center, y_center

def draw_ball(input_frame, ult_result: ultralytics.engine.results.Results, ball_class_index: int = 0):
    frame = input_frame.copy()
    x, y = get_ball_xy(ult_result, ball_class_index)
    if x is None or y is None:
        return frame
    cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), 2)
    return frame

def top_confidence_indexes(conf: list[float], how_many: int) -> tuple[int]:
    """Get top 'how_many' items of the list l. Returns tuple of indexes."""
    return sorted(range(len(conf)), key=lambda i: conf[i], reverse=True)[:how_many]


def conf_boxes_reduce(conf: list[float], boxes: list[np.ndarray], how_many: int) -> tuple[list[float], list[np.ndarray]]:
    """Put a ceiling on maximum number of bounding boxes, prioritize boxes with higher confidence."""
    if len(conf) > 0:
        top_indexes = top_confidence_indexes(conf, how_many=how_many)
        result_conf = [conf[index] for index in top_indexes]
        result_boxes = [boxes[index] for index in top_indexes]
        return result_conf, result_boxes
    else:
        return [], []

def postprocess_reduce(cls: list[int], conf: list[float], boxes: ultralytics.engine.results.Boxes, cls_to_name: dict[int, str]) -> tuple[list[int], list[float], np.array]:
    """Reduce the bounding box number according to the following rules.
    - Max 1 best ball prediction
    - Max 2 best paddle predictions
    - Max 1 scorekeeper prediciton
    - Max 2 player predictions
    """
    name_to_cls = {name: index for index, name in cls_to_name.items()}
    maxes = {"player": 2, "scorekeeper": 1, "ball": 1, "paddle": 2}  # ordered by avg box size for ease of use in LS

    result_conf = []
    result_boxes = []
    result_cls = []
    for label, maximum in maxes.items():
        # Select confidences and bboxes based on label
        label_conf = [conf[i] for i in range(len(cls)) if cls_to_name[cls[i]] == label]
        label_boxes = [boxes[i] for i in range(len(cls)) if cls_to_name[cls[i]] == label]

        # Reduce
        label_conf, label_boxes = conf_boxes_reduce(label_conf, label_boxes, maximum)
        result_cls += [name_to_cls[label]] * len(label_conf)
        result_boxes += label_boxes
        result_conf += label_conf
    
    return result_cls, result_conf, result_boxes

def draw_predictions(input_frame, ult_result: ultralytics.engine.results.Results):
    frame = input_frame.copy()
    cls = ult_result.boxes.cls.cpu().numpy().astype(int)
    conf = ult_result.boxes.conf.cpu().numpy().astype(float)
    boxes = ult_result.boxes
    cls_to_name = {index: name for index, name in ult_result.names.items()}
    cls, conf, boxes = postprocess_reduce(cls, conf, boxes, cls_to_name)
    colors = {"ball": (100, 100, 255), "player": (0, 255, 0), "scorekeeper": (200, 0, 200), "paddle": (255, 100, 0)}
    for i in range(len(cls)):
        class_index = cls[i]
        x_center, y_center, width, height = boxes[i].xywh.cpu().numpy().astype(float)[0].tolist()
        x_center = int(x_center)
        y_center = int(y_center)
        width = int(width)
        height = int(height)
        color = colors[cls_to_name[class_index]]
        cv2.rectangle(frame, (x_center - width // 2, y_center - height // 2), (x_center + width // 2, y_center + height // 2), color, 2)
        cv2.putText(frame, cls_to_name[class_index], (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame



def main():
    # Load the YOLOv8 model
    MODEL_WEIGHTS = 'data/models/blurry/yolov8x__TRAIN42min_1000random__VAL2hrs_30fps_1to2min_0.83x_speed__TEST42min_100random__222epochs_earlystopping/weights/best.pt'
    ON_ANNOTATED_IMAGES = True
    TRACK = False  # works well with high framerate
    # Save video to file
    out = None
    #out = cv2.VideoWriter('2yolov8x__TRAIN42min_1000random__VAL2hrs_30fps_1to2min_0.83x_speed__TEST42min_100random__222epochs_earlystopping.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 120, (1920, 1080))
    
    model = YOLO(MODEL_WEIGHTS, task="detect")
    if ON_ANNOTATED_IMAGES:

        YOLO_DATA_DIR = r"data\yolo_datasets\blurry\42min_1000random".replace("\\", "/")
        annotations_dir = YOLO_DATA_DIR + "/labels"
        images_dir = YOLO_DATA_DIR + "/images"
        classes_dir = YOLO_DATA_DIR + "/classes.txt"

        label_count = len([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
        images_generator = generate_images(images_dir)
        annotations_generator = generate_annotations(annotations_dir)
        classes = load_classes(classes_dir)

        for frame, annotation in tqdm(zip(images_generator, annotations_generator), total=label_count):
            # Inference
            if TRACK:
                result = model.track(frame, verbose=False)[0]  # define what classes to predict
            else:
                result = model.predict(frame, verbose=False)[0]  # define what classes to predict
            
            # Annotations
            frame = draw_annotations(frame, annotation, classes, line_width=5, opacity=0.4)

            # Predictions
            frame = draw_predictions(frame, result)

            display_frame(frame, out, delay=1)
    else:
        VIDEO_PATH = r"C:\Users\yannick.gibson\projects\work\important\ball-tracker\data\videos\blurry\42min.mp4"
        SHOW_EVERY = 10

        # Open the video file
        cap = cv2.VideoCapture(VIDEO_PATH)
        i = 0
        
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if i <= 1200:
                print("skipping: ", i)
            elif success:
                if i % SHOW_EVERY == 0:
                    if TRACK:
                        result = model.track(frame, verbose=False, tracker="bytetrack.yaml")[0]  # define what classes to predict
                        frame = result.plot()
                        frame = draw_ball(frame, result)
                    else:
                        result = model.predict(frame, verbose=False, conf=0.3)[0]  # define what classes to predict
                        frame = draw_ball(frame, result)
                        frame = draw_predictions(frame, result)
                    display_frame(frame, out)
                    print(f"frame: {i}")
            i += 1
        else:
            print("Could not open video")


if __name__ == "__main__":
    main()
