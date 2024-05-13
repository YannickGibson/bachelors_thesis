"""This module is used for fine-tuning a YOLO model."""

import argparse
from ultralytics import YOLO

DATA_PATH_DEFAULT = r'C:\Users\yannick.gibson\projects\school\BP\bachelors_thesis\annotation\mydata\yolo_datasets\blurry\42min_100random\\data.yaml'
EPOCHS_DEFAULT = 10
MODEL_DEFAULT = "yolov8n.pt"


def parse_arguments() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the training data", nargs='?', default=DATA_PATH_DEFAULT, type=str)
    parser.add_argument("--epochs", help="Number of epochs to train the model", nargs='?', default=EPOCHS_DEFAULT, type=int)
    parser.add_argument("--model_name", help="Name of the model, for example: yolov8n.pt", nargs='?', default=MODEL_DEFAULT, type=str)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_arguments()
    # Train the model
    model = YOLO(args.model_name)
    model.train(data=args.data_path, epochs=args.epochs, plots=True, project="data/models/runs")


if __name__ == "__main__":
    main()

