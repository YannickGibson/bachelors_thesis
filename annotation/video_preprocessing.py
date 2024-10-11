"""Preprocess video before giving it to label studio pipeline"""

import cv2
from tqdm import tqdm
import os
import numpy as np
import subprocess
import skvideo.io


def trim_video(input_file: str, output_file: str, start_time: float, end_time: float) -> None:
    """Use FFMPEG to trim video from start_time to end_time in seconds.
    Args:
        input_file: Path to the input video file.
        output_file: Path to the output video file.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        """
    ffmpeg_cmd = [
        'ffmpeg',
        '-ss', str(start_time),  # Start time in seconds
        '-i', input_file,
        '-t', str(end_time),    # End time in seconds
        '-c', 'copy',           # Copy the codec
        output_file,
        "-y"                    # Accept all prompts
    ]

    # Run FFmpeg command in a subprocess and await its completion
    with subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE) as process:
        process.wait()


def trim_and_alter_fps(input_path: str, output_path: str = None,
                    target_fps: int = 30, play_fps: int = None, minutes_from: int = 0, minutes_to: float = float("inf")) -> None:
    """Trim video from minutes_from to minutes_to, select nth frame based on target_fps. Output video with play_fps set to target video fps.
    Example: video input fps=60, target_fps=30, play_fps=15.
        Output video will be created from every second frame (60/30=2) from the input video.
        Resulting speed will be .5x, corresponding to 15 FPS.
    Args:
        input_path: Path to the input video file.
        output_path: Path to the output video file. If None, it will be created based on input filename.
        target_fps: Target frames per second to output.
        play_fps: Frames per second to play the video. If None, it will be set to target_fps.
        minutes_from: Start time in minutes.
        minutes_to: End time in minutes.
    """
    trim_video(input_path, "trimmed_temp.ts", start_time=int(minutes_from * 60), end_time=int(minutes_to * 60 - minutes_from * 60))
    
    # Open trimmed video
    cap = cv2.VideoCapture("trimmed_temp.ts")
    if not cap.isOpened():
        raise ValueError(f"Video file at {input_path} could not be opened. Temporary file might not have been deleted.")
    
    # Check FPS
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    if target_fps > input_fps:
        raise ValueError("Target FPS cannot be greater than input FPS.")
    if input_fps % target_fps != 0:
        raise ValueError("Target FPS must be a factor of input FPS to selectively pick frames. Got {input_fps} / {target_fps} is not an int.")

    # Print video information
    extension = input_path.split(".")[-1]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    select_every = input_fps // target_fps
    input_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    start_frame = int(minutes_from * 60 * input_fps)
    end_frame = int(minutes_to * 60 * input_fps)
    print(f"Input FPS: {input_fps:>3}, Input frame count: {total_frames:>6}, Input extension: {'.' + extension:>5}")
    print(f"Output FPS: {target_fps:>3}, Output frame count {total_frames // select_every:>6}, Output extension: .mp4")
    print(f"Output Video length: {minutes_to - minutes_from} minute(s) | Video Dimensions: {input_size} px")
    print(f"Start frame: {start_frame}, End frame: {end_frame}")
    
    # Set playback speed
    if play_fps is None:
        play_fps = target_fps
    playback_speed = play_fps / target_fps

    # Set output name
    if output_path is None:
        input_filename = input_path.split("/")[-1].split("\\")[-1].split(".")[0]
        output_path = f"{input_filename}_{target_fps}fps_{minutes_from}to{minutes_to}min_{playback_speed:.2f}x_speed.mp4"

    # Selectively read and write video frames
    writer = skvideo.io.FFmpegWriter(output_path, inputdict={"-r": f"{play_fps}"})
    print(f"Converting .ts to .mp4 and selecting every {select_every}th frame - {output_path}.")
    for frame_index in tqdm(range(0, total_frames)):
        # Load frame
        ret, bgr_frame = cap.read()
        if not ret:
            print("Exiting prematurely - Frame with absolute positon of {frame_index} could not be read.")
            break

        if frame_index % select_every == 0:
            # Write frame
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            writer.writeFrame(rgb_frame)

    # Close video writer and release video capture
    writer.close()
    cap.release()

    print(f"Video was saved to {output_path}.")


def grab_frames_at_random(input_video: str, output_folder: str, num_frames: int, overwrite_output_folder: bool = False) -> None:
    """
    Grab random frames from video and save them to output folder.

    Args:
        input_video: Input video file path.
        output_folder: Output folder path.
        num_frames: Number of frames to grab.
        clean_output_folder: If running twice with same output_folder name, use this option.
    """
    # Open video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Video file at {input_video} could not be opened.")
    
    # Check if output folder exists
    if not os.path.exists(output_folder):
        inp = input(f"Output folder at '{output_folder}' does not exist. Do you want to create it? ((y,yes)/(anything else)): ")
        if inp.lower() in ["y", "yes"]: 
            os.makedirs(output_folder)
        else:
            raise ValueError("Output folder does not exist. Please create it.")
        

    # Clean output folder
    # Get frames in output folder
    frame_files = []
    for file in os.listdir(output_folder):
        if file.startswith("frame_"):  # Only delete files with prefix "frame_"
            frame_files.append(file)
        if overwrite_output_folder:
            if len(frame_files) > 0:
                print(f"Deleting {len(frame_files)} files from output folder.")
                for file in tqdm(frame_files):
                    os.remove(os.path.join(output_folder, file))
            else:
                raise ValueError(f"Output folder has frames in it ({len(frame_files)} frames).")

    # Generate randomized indexes
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rand_indexes = np.random.choice(total_frames, num_frames, replace=False)  # replace=False means no duplicates
    zero_pad = len(str(total_frames))

    # Save frames at selected indexes
    print(f"Saving {num_frames} frames to {output_folder}.")
    for i in tqdm(range(num_frames)):
        # Jump to specific frame
        frame_index = rand_indexes[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read video
        ret, frame = cap.read()
        if not ret:
            print(f"Frame with absolute positon of {frame_index} could not be read.")
            continue
        
        cv2.imwrite(fr"{output_folder}\frame_{frame_index:0{zero_pad}}.jpg", frame)


def main() -> None:
    case = 2
    if case == 0:
        # Trim only
        input_video = r"C:\Users\yannick.gibson\projects\work\important\ball-tracker\videos\blurred\2hrs.ts"
        output_video = 'output_trimmed.ts'
        start_time_seconds = 1000
        end_time_seconds = 1020
        trim_video(input_video, output_video, start_time_seconds, end_time_seconds)
    elif case == 1:
        trim_and_alter_fps(
            input_path=r"C:\Users\yannick.gibson\projects\work\important\ball-tracker\videos\blurred\2hrs.ts",
            output_path=None,
            minutes_from=2,
            minutes_to=3,
            target_fps=24
        )
    elif case == 2:
        grab_frames_at_random(
            input_video=os.environ["RANDOM_VIDEO_PATH"],
            output_folder=os.environ["RANDOM_OUTPUT_FOLDER"],
            num_frames=2000,
            overwrite_output_folder=False
        )


if __name__ == "__main__":
    main()
