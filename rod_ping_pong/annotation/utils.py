"""Utilities for annotation"""

import os
import cv2
from typing import Iterator
import numpy as np


global_paused = False


def generate_annotations(annotations_path: str) -> Iterator[list]:
    """Yield annotation in a list from annotations path"""
    onlyfiles = [f for f in os.listdir(annotations_path) if os.path.isfile(os.path.join(annotations_path, f))]
    for file in onlyfiles:
        with open(annotations_path + "\\" + file, 'r') as f:
            s = f.read().replace("\n\n", "\n")
            annotation = s.split("\n")

        if annotation[-1] == "":
            yield annotation[:-1]  # ommit last one, because it is an empty string
        else:
            yield annotation


def generate_images(images_path: str) -> Iterator[np.array]:
    """Yield images in a list from images path"""
    onlyfiles = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
    for file in onlyfiles:
        img = cv2.imread(images_path + "\\" + file, cv2.IMREAD_COLOR)
        yield img


def load_classes(classes_path: str) -> list:
    """Return class names in a list from classes.txt path"""
    with open(classes_path, 'r') as f:
        classes = f.read().split("\n")
    return classes


def display_frame(frame: np.ndarray, out: cv2.VideoCapture = None, delay: int = 1) -> None:
    """Display frame using cv2 with no delay, exit using 'q' key, pause using 'p' key, and forward one frame using 'e' key."""
    global global_paused
    cv2.imshow("video", frame)
    if out is not None:
        out.write(frame)
    key = cv2.waitKey(delay)
    if key & 0xFF == ord('p') or global_paused:
        global_paused = True
        while True:
            key = cv2.waitKey(delay)
            if key & 0xFF == ord('p'):
                global_paused = False
                break
            elif key & 0xFF == ord('e'):  # forward one frame
                break
            elif key & 0xFF == ord('q'):
                exit(0)
    if key & 0xFF == ord('q'):
        exit(0)
