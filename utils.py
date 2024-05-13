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


def draw_ball(frame: np.ndarray, x: float, y: float, bgr_color: tuple[int, int, int] = (0, 0, 255), radius: int = 5) -> np.ndarray:
    """Display frame using frame number"""
    result_frame = frame.copy()
    cv2.circle(result_frame, (round(x), round(y)), radius, bgr_color, thickness=5)
    return result_frame