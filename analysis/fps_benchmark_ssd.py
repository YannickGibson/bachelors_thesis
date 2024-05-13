"""This script is used to calculate average fps for a SSD model"""

import torch
import torchvision
from torchvision import transforms as T
import cv2

from utils import Stopwatch


def main() -> None:
    AVERAGE_FPS_ITERATIONS = 10
    VIDEO_PATH = r"C:\Users\yannick.gibson\projects\work\important\ball-tracker\data\videos\blurry\42min.mp4"

    # Setup SSD model
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    model: torchvision.models.detection.SSD = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    model.to(device)
    model.eval()  # Training mode

    # Prepare variables
    cap = cv2.VideoCapture(VIDEO_PATH)
    transform = T.ToTensor()
    stopwatch: Stopwatch = Stopwatch()
    iteration_count = 1

    # Inference
    while cap.isOpened():
        success, frame = cap.read()
        img = transform(frame)
        if success:
            with stopwatch:
                with torch.no_grad():
                    _ = model([img])

            if iteration_count % AVERAGE_FPS_ITERATIONS == 0:
                print(f"Cummulative Average FPS (out of {AVERAGE_FPS_ITERATIONS}): {round(stopwatch.get_avg_cum_fps(reset_cum_elapsed=True))}")
            iteration_count += 1

    print("Exiting...")

if __name__ == "__main__":
    main()
