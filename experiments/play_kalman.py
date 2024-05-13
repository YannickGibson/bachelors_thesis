"""Utilizing kalman filter with YOLOv8 ball detections."""

import pandas as pd
import cv2
from tqdm import tqdm

from utils import display_frame, draw_ball
from kalman_filter import KalmanFilter


def main() -> None:
    cap = cv2.VideoCapture(r"data/videos/blurred/42min_120fps.mp4")
    if not cap.isOpened():
        raise ValueError("Video not found")

    # export video to mp4
    out = None
    #out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))

    # Load and prepare the data
    df = pd.read_csv("data/ball_positions_yolov8n_42min_120fps.txt", delimiter=";", header=0)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    df["certainty"] = df["certainty"].astype(float)
    df["frame_number"] = df["frame_number"].astype(int)


    # Initialize the Kalman Filter
    KF = KalmanFilter(
        dt=0.1,          # Sets the time step
        u_x=0,           # Acceleration in the x-axis
        u_y=19.81,       # Acceleration in the y-axis
        std_acc=3,       # Standard deviation of the acceleration noise
        x_std_meas=0.3,  # Standard deviation of the noise measurement in the x-axis
        y_std_meas=0.3   # Standard deviation of the noise measurement in the y-axis
    ) 
    current_frame_number = df["frame_number"].iloc[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)
    print(f"Video frame count: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    print(f"Starting frame number: {current_frame_number}")
    MAX_IMPROVIZATION = 4  # numbers of frames to trust kalman.predict
    improvization_count = 0
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        df_frame_number = row["frame_number"]
        while True:
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Frame not found")
            
            if improvization_count < MAX_IMPROVIZATION:
                (kalman_predicted_x, kalman_predicted_y) = map(int, KF.predict())
            if current_frame_number == df_frame_number:  # Model has predicted
                improvization_count = 0
                break
            elif current_frame_number < df_frame_number:  # Model has not predicted yet
                if improvization_count < MAX_IMPROVIZATION:
                    frame = draw_ball(frame, kalman_predicted_x, kalman_predicted_y, bgr_color=(0, 255, 128))
                display_frame(frame, out)
                current_frame_number += 1
                improvization_count += 1

        if row["certainty"] > 0.3:
            # Display model result
            model_x, model_y = row["x"], row["y"]
            kalman_estimated_x, kalman_estimated_y = KF.update([[model_x], [model_y]])
            kalman_estimated_x, kalman_estimated_y = int(kalman_estimated_x), int(kalman_estimated_y)
            frame = draw_ball(frame, kalman_estimated_x, kalman_estimated_y, bgr_color=(255, 0, 0))
            frame = draw_ball(frame, model_x, model_y, radius=2)
        else:
            # Use Kalman if model is not certain enough
            if improvization_count < MAX_IMPROVIZATION:
                (kalman_predicted_x, kalman_predicted_y) = map(int, KF.predict())
                frame = draw_ball(frame, kalman_predicted_x, kalman_predicted_y, bgr_color=(0, 255, 128))
        
        display_frame(frame, out)
        current_frame_number += 1


if __name__ == "__main__":
    main()
