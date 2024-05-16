"""Utilities common for the whole repository."""

import time
from typing import Any, Self
import numpy as np
import os
import cv2
from typing import Iterator


global_paused = False


class Stopwatch:
    """Uses context manager to measure time for a block of code."""

    def __init__(self) -> None:
        """Initialize the class."""
        self.start_time = 0.0
        self.total_iterations = 0
        self.total_elapsed = 0.0
        self.elapsed = 0.0
        self.cumulative_elapsed = 0.0
        self.cumulative_iterations = 0

    def __enter__(self) -> Self:
        """Start the stopwatch.

        Returns:
            self
        """
        self.start_time = time.time()
        return self

    def __exit__(self, *args: Any) -> None:  # noqa D105
        """Stop the stopwatch and save the elapsed time."""
        self.elapsed = time.time() - self.start_time
        self.total_elapsed += self.elapsed
        self.cumulative_elapsed += self.elapsed
        self.cumulative_iterations += 1
        self.total_iterations += 1

    @property
    def fps(self) -> float:
        """Return frames per second of last iteration."""
        if self.elapsed == 0:
            return np.nan
        return 1 / (self.elapsed)

    def get_avg_cum_time(self, reset_cum_elapsed: bool = False) -> float:
        """Return average time elapsed in seconds for iterations that were called after the last cumulative reset.

        Args:
            reset_cum_elapsed: whether to reset cumulative elapsed time in the stopwatch.

        Returns:
            average tracked time since last reset.
        """
        res = 0.0 if self.cumulative_iterations == 0 else self.cumulative_elapsed / self.cumulative_iterations

        if reset_cum_elapsed:
            self.reset_cum_elapsed()
        return res

    def get_avg_cum_fps(self, reset_cum_elapsed: bool = False) -> float:
        """Return average frames per second since last reset.

        Args:
            reset_cum_elapsed: boolean denoting whether to reset the cumulative time in the stopwatch.

        Returns:
            frames per second since last reset.
        """
        avg_time = self.get_avg_cum_time(reset_cum_elapsed)
        if avg_time == 0:
            return np.nan
        return 1 / avg_time

    def reset_cum_elapsed(self) -> None:
        """Reset cumulative time in the stopwatch."""
        self.cumulative_elapsed = 0
        self.cumulative_iterations = 0


def load_annotation(annotation_path: str) -> list:
    """Load an annotation from defined path with an appropriate format."""
    with open(annotation_path, 'r') as f:
        s = f.read().replace("\n\n", "\n")
        annotation = s.split("\n")

    if annotation[-1] == "":
        res = [a.split(" ") for a in annotation[:-1]]
    else:
        res = [a.split(" ") for a in annotation]

    # Convert all values from string to a number
    for i in range(len(res)):
        res[i][0] = int(res[i][0])
        res[i][1] = float(res[i][1])
        res[i][2] = float(res[i][2])
        res[i][3] = float(res[i][3])
        res[i][4] = float(res[i][4])
    return res

def generate_annotations(annotations_path: str) -> Iterator[list]:
    """Yield annotation in a list from annotations path"""
    onlyfiles = [f for f in os.listdir(annotations_path) if os.path.isfile(os.path.join(annotations_path, f))]
    for file in onlyfiles:
        yield load_annotation(annotations_path + "\\" + file)


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


def draw_ball(frame: np.ndarray, x: float, y: float, bgr_color: tuple[int, int, int] = (0, 0, 255), radius: int = 5) -> np.ndarray:
    """Display frame using frame number"""
    result_frame = frame.copy()
    cv2.circle(result_frame, (round(x), round(y)), radius, bgr_color, thickness=5)
    return result_frame


def draw_annotations(input_frame, annotations, classes, line_width=2, opacity=1, color=(0, 255, 0)):
    frame = input_frame.copy()
    for annotation in annotations:
        # Split the annotation into its components
        class_index = int(annotation[0])
        x_center = int(float(annotation[1]) * frame.shape[1])
        y_center = int(float(annotation[2]) * frame.shape[0])
        width = int(float(annotation[3]) * frame.shape[1])
        height = int(float(annotation[4]) * frame.shape[0])

        # Draw the bounding box
        cv2.rectangle(frame, (x_center - width // 2, y_center - height // 2), (x_center + width // 2, y_center + height // 2), color, line_width)

        # Draw the class name
        cv2.putText(frame, classes[class_index], (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
    # Change opacity of annotations
    frame = cv2.addWeighted(frame, opacity, input_frame, 1 - opacity, 0)
    return frame
