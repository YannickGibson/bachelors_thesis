"""Utilities for annotation"""

import os
import cv2
from typing import Iterator
import numpy as np


def generate_annotations(annotations_path: str) -> Iterator[list]:
    # "listdir" provides sorted file names
    onlyfiles = [f for f in os.listdir(annotations_path) if os.path.isfile(os.path.join(annotations_path, f))]
    for file in onlyfiles:
        with open(annotations_path + "\\" + file, 'r') as f:
            annotation = f.read().split("\n\n")

        yield annotation[:-1]  # ommit last one, because it is an empty string

def generate_images(images_path: str) -> Iterator[np.array]:
    # "listdir" provides sorted file names
    onlyfiles = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
    for file in onlyfiles:
        img = cv2.imread(images_path + "\\" + file, cv2.IMREAD_COLOR)
        yield img

def load_classes(classes_path: str) -> list:
    with open(classes_path, 'r') as f:
        classes = f.read().split("\n")
    return classes
