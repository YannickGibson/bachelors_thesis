import cv2
import os
from tqdm import tqdm
import sys


def crop(video_path, top_crop: int = 0, bottom_crop: int = 0, left_crop: int = 0, right_crop: int = 0) -> None:
    """Crop the desired video and save it with name output.<extension>"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Video file at {video_path} could not be opened.")
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc_str = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, byteorder=sys.byteorder).decode()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    extension = video_path.split(".")[-1]
    output_path = f'output.{extension}'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width - left_crop - right_crop, height - top_crop - bottom_crop))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    

    print(f"Cropping video to '{output_path}'")
    for _ in tqdm(range(frame_count)):
        success, frame = cap.read()
        if success:
            out.write(frame[top_crop:height - bottom_crop, left_crop:width - right_crop])
        else:
            break



if __name__ == "__main__":
    crop(
        video_path=os.environ["CROP_VIDEO_PATH"],
        top_crop=int(os.environ["CROP_TOP_CROP"]), 
        bottom_crop=int(os.environ["CROP_BOTTOM_CROP"]), 
        left_crop=int(os.environ["CROP_LEFT_CROP"]), 
        right_crop=int(os.environ["CROP_RIGHT_CROP"])
    )
