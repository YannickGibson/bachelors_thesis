"""Generating pre-annotatinos using model inference and saving them in Label Studio format."""

import json
import os
import cv2
from typing import Iterator
import numpy as np
import ultralytics as ult
from tqdm import tqdm

MYDATA_PATH = os.environ["PREANNOTATION_MYDATA_PATH"].replace("\\", "/")
RELATIVE_RESULTS_FOLDER_PATH = "results"
BASE_RESULTS_NAME = "preannotation"

__folder__ = os.path.dirname(__file__)
BLANK_RESULT = os.path.join(__folder__, 'blank_result.json')
BLANK_TASK = os.path.join(__folder__, 'blank_task.json')


def get_task(ls_url, model_version) -> dict:
    """Generates task dictionary from blank task template"""

    with open(BLANK_TASK) as f:
        task = json.load(f)

    task["data"]["image"] = ls_url
    task["predictions"][0]["model_version"] = model_version

    return task


def get_result(xn: int, yn: int, widthn: int, heightn: int, label: str, id: str) -> dict:  
    """Adds parameters to label studio result using template file."""

    with open(BLANK_RESULT) as f:
        result = json.load(f)

    result["value"]["x"] = (xn - widthn / 2) * 100
    result["value"]["y"] = (yn - heightn / 2) * 100
    result["value"]["width"] = widthn * 100
    result["value"]["height"] = heightn * 100
    result["value"]["rectanglelabels"][0] = label
    result["id"] = id

    return result


def image_generator(relative_images_folder_path: str, limit: int | None = None) -> Iterator[tuple[np.array, str]]:
    """Generate images from folder path in alphabetical order with their names.
    All files in folder have to be images.
    """
    images_folder_path = os.path.join(MYDATA_PATH, relative_images_folder_path).replace("\\", "/")
    onlyfiles = [f for f in os.listdir(images_folder_path) if os.path.isfile(os.path.join(images_folder_path, f))]
    if limit is not None:
        onlyfiles = onlyfiles[:limit]
    for image_name in tqdm(onlyfiles):
        image_path = os.path.join(images_folder_path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        yield image, image_name


def save_preannotation(list_of_tasks: list[dict], absolute_save_folder_path: str) -> None:
    """Create index from 0 to inifinity choosing lowest possible number that is not taken in the folder RELATIVE_RESULTS_FOLDER_PATH"""
    index = 0
    while os.path.exists(path := os.path.join(absolute_save_folder_path, f"{BASE_RESULTS_NAME}_{index}.json").replace("\\", "/")):
        index += 1
    
    with open(path, "w") as f:
        json.dump(list_of_tasks, f, indent=4)


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


def postprocess_reduce(cls: list[int], conf: list[float], boxes: ult.engine.results.Boxes, cls_to_name: dict[int, str]) -> tuple[list[int], list[float], np.array]:
    """Reduce the bounding box number according to the following rules.
    - Max 1 best ball prediction
    """
    name_to_cls = {name: index for index, name in cls_to_name.items()}
    maxes = {"ball": 1}  # Alow maximum of 1 ball prediction per frame

    result_conf = []
    result_boxes = []
    result_cls = []
    for label, maximum in maxes.items():
        # Select confidences and bboxes based on label
        label_conf = [conf[i] for i in range(len(cls)) if cls_to_name[cls[i]] == label]
        label_boxes = [boxes[i] for i in range(len(cls)) if cls_to_name[cls[i]] == label]

        # Reduce
        result_cls += [name_to_cls[label]] * len(label_conf)
        label_conf, label_boxes = conf_boxes_reduce(label_conf, label_boxes, maximum)
        result_boxes += label_boxes
        result_conf += label_conf
    
    return result_cls, result_conf, result_boxes


def main() -> None:
    # Configurable constants
    MODEL_PATH = os.environ["PREANNOTATION_MODEL_PATH"].replace("\\", "/")
    MODEL_VERSION = os.environ["PREANNOTATION_MODEL_VERSION"]
    RELATIVE_IMAGES_FOLDER_PATH = os.environ["PREANNOTATION_RELATIVE_IMAGES_FOLDER_PATH"].replace("\\", "/")
    LIMIT = int(os.environ["PREANNOTATION_LIMIT"]) if "PREANNOTATION_LIMIT" in os.environ else None
    if MODEL_VERSION not in MODEL_PATH:
        raise ValueError("Model version is not in model path. Did you forget to update the model version?")
    IMAGES_FOLDER_NAME = os.path.basename(RELATIVE_IMAGES_FOLDER_PATH)
    
    # Load model
    model = ult.YOLO(MODEL_PATH)
    
    # Iterate over images
    list_of_tasks = []
    for image, image_name in image_generator(RELATIVE_IMAGES_FOLDER_PATH, LIMIT):
        # Inference
        ult_result: ult.engine.results.Results = model.predict(image, verbose=False)[0]

        # Create a task
        ls_image_path = os.path.join(RELATIVE_IMAGES_FOLDER_PATH, image_name).replace("\\", "/")
        ls_url = f"/data/local-files/?d={ls_image_path}"
        task = get_task(ls_url, MODEL_VERSION)


        # Reduce bboxes according to postprocess rules
        cls = ult_result.boxes.cls.cpu().numpy().astype(int)
        conf = ult_result.boxes.conf.cpu().numpy().astype(float)
        ult_result_boxes = ult_result.boxes
        cls, conf, ult_result_boxes = postprocess_reduce(cls, conf, ult_result_boxes, ult_result.names)

        # Results to label studio (ls) format
        for i, box in enumerate(ult_result_boxes):
            xn, yn, wn, hn = box.xywhn.cpu().numpy().astype(float)[0].tolist()
            label = ult_result.names[cls[i]]
            box_id = f"{image_name}_{i}"
            ls_result = get_result(xn, yn, wn, hn, label, box_id)
            task["predictions"][0]["result"].append(ls_result)
        
        list_of_tasks.append(task)

    # Save pre-annotation
    absolute_save_folder_path = os.path.join(__folder__, RELATIVE_RESULTS_FOLDER_PATH, IMAGES_FOLDER_NAME, MODEL_VERSION).replace("\\", "/")
    if not os.path.exists(absolute_save_folder_path):
        print(f"Creating folder: {absolute_save_folder_path}")
        os.makedirs(absolute_save_folder_path)
    save_preannotation(list_of_tasks, absolute_save_folder_path)


if __name__ == "__main__":
    main()
