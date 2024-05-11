"""This module is used for fine-tuning a YOLO model."""

import argparse
from ultralytics import YOLO

DATA_PATH_DEFAULT = r'C:\Users\matematik\projects\yannick\yolo_training\training_data\ping_05_cam_2_30fps_0to1min\data_autosplit.yaml'
EPOCHS_DEFAULT = 10


def parse_arguments() -> tuple[str, int]:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to the training data", default=DATA_PATH_DEFAULT)
    parser.add_argument("epochs", help="Number of epochs to train the model", default=EPOCHS_DEFAULT)
    args = parser.parse_args()
    DATA_PATH = args.data_path
    EPOCHS = args.epochs
    return DATA_PATH, EPOCHS


if __name__ == "__main__":
    DATA_PATH, EPOCHS = parse_arguments()

    # Train the model
    model = YOLO("yolov8n.pt")
    model.train(data=DATA_PATH, epochs=EPOCHS, plots=True)
