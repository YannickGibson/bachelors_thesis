"""This script is used to calculate average fps for a specified yolo model"""

import cv2
from ultralytics import YOLO
import torch

from utils import Stopwatch


def main() -> None:
    MODEL_WEIGHTS = r""
    VIDEO_PATH = r""
    AVERAGE_FPS_ITERATIONS = 10

    # Checks
    if not torch.cuda.is_available():
        raise Exception("CUDA is not available. Exiting...")
    else:
        print("CUDA is available")

    # Prepare variables
    model: YOLO = YOLO(MODEL_WEIGHTS).to("cuda")
    cap = cv2.VideoCapture(VIDEO_PATH)
    stopwatch: Stopwatch = Stopwatch()
    iteration_count = 1

    # Inference
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            with stopwatch:
                _ = model(frame, verbose=False)

            if iteration_count % AVERAGE_FPS_ITERATIONS == 0:
                print(f"Cummulative Average FPS (out of {AVERAGE_FPS_ITERATIONS}): {round(stopwatch.get_avg_cum_fps(reset_cum_elapsed=True))}")
            iteration_count += 1

    print("Exiting...")

if __name__ == "__main__":
    main()
