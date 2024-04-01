"""Preprocess video before giving it to label studio pipeline"""

import cv2
from tqdm import tqdm
import os
import numpy as np
import subprocess


def trim_video(input_file, output_file, start_time, end_time):
    # FFmpeg command to trim the video
    ffmpeg_cmd = [
        'ffmpeg',
        '-ss', str(start_time),  # Start time in seconds
        '-i', input_file,
        '-t', str(end_time),    # End time in seconds
        '-c', 'copy',            # Copy the codec
        output_file,
        "-y"
    ]

    # Run FFmpeg command
    with subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE) as process:
        process.wait() # Wait for the process to finish


def preprocess_video(input_path, output_path = None, 
                    target_fps=30, minutes_from=0, minutes_to=float("inf")):

    print( int(minutes_from * 60), int(minutes_to * 60 - minutes_from * 60))
    trim_video(input_path, "trimmed_temp.ts", int(minutes_from * 60), int(minutes_to * 60 - minutes_from * 60))
    
    cap = cv2.VideoCapture("trimmed_temp.ts")

    if not cap.isOpened():
        raise ValueError(f"Video file at {input_path} could not be opened.")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = total_frames / fps
    if target_fps > fps:
        raise ValueError("Target FPS cannot be greater than input FPS.")
    if fps % target_fps != 0:
        raise ValueError("Target FPS must be a factor of input FPS. Got {fps} / {target_fps} is not an int.")
    #if minutes_to == float("inf"):
    #    minutes_to = total_seconds // 60
    #elif minutes_to < 0 or minutes_to > total_seconds // 60:
    #    raise ValueError(f"Argument minutes_to must be in the interval [0, {total_seconds // 60}]. Got {minutes_to}.")        
    select_every = fps // target_fps
    extension = input_path.split(".")[-1]

    input_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"Input FPS: {fps:>3}, Input frame count: {total_frames:>6}, Input extension: {'.' + extension:>5}")
    print(f"Output FPS: {target_fps:>3}, Output frame count {total_frames // select_every:>6}, Output extension: .mp4")
    print(f"Output Video length: {minutes_to - minutes_from} minute(s) | Video Dimensions: {input_size} px")
    

    start_frame = int(minutes_from * 60 * fps)
    end_frame = int(minutes_to * 60 * fps)
    print(f"Start frame: {start_frame}, End frame: {end_frame}")
    #cap.set(cv2.CAP_PROP_POS_MSEC, minutes_from * 60 * 1000)  # set start frame

    input_filename = input_path.split("/")[-1].split("\\")[-1].split(".")[0]
    #if output_path is None:
    #    output_path = f"{input_filename}_{target_fps}fps_{minutes_from}to{minutes_to}min.mp4"
    #out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, input_size)

    
    import skvideo.io

    play_fps = 24
    playback_speed = play_fps / target_fps

    if output_path is None:
        output_path = f"{input_filename}_{target_fps}fps_{minutes_from}to{minutes_to}min_{playback_speed:.2f}x_speed.mp4"
    writer = skvideo.io.FFmpegWriter(output_path, inputdict={"-r": f"{play_fps}"})

    # Read and write select_every'th frame
    #for frame_index in tqdm(range(start_frame, min(end_frame, total_frames))):
    print(f"Converting .ts to .mp4 and selecting every {select_every}th frame - {output_path}.")
    for frame_index in tqdm(range(0, total_frames)):
        ret, frame = cap.read()
        if not ret:
            print("Exiting prematurely - Frame with absolute positon of {frame_index} could not be read.")
            break

        if frame_index % select_every == 0:
                #cv2.imshow("frame", frame)
                ## wait for key press to continue showing next frame
                #if cv2.waitKey(330) == ord('q'):
                #    continue
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                writer.writeFrame(rgb_frame)
                #out.write(frame)
    
    writer.close()
    cap.release()
    #out.release()
    #os.remove("trimmed_temp.ts")
    print(f"Video was saved to {output_path}.")


def grab_frames_at_random(input_video, output_folder, num_frames, clean_output_folder = False):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Video file at {input_video} could not be opened.")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    rand_indexes = np.random.choice(total_frames, num_frames, replace=False)  # replace=False means no duplicates 
    print("Generating frames to folder: ", output_folder)

    # delete every file in output folder with prefix frame_
    if clean_output_folder:
        for file in os.listdir(output_folder):
            if file.startswith("frame_"):
                os.remove(os.path.join(output_folder, file))

    for i in range(num_frames):
        frame_index = rand_indexes[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(f"Frame with absolute positon of {frame_index} could not be read.")
            continue
        cv2.imwrite(fr"{output_folder}\frame_{frame_index}.jpg", frame)    


if __name__ == "__main__":

    # input_video = r"C:\Users\yannick.gibson\projects\work\important\ball-tracker\videos\blurred\2hrs.ts"
    # output_video = 'output_trimmed.ts'
    # start_time_seconds = 1000
    # end_time_seconds = 1020
    # trim_video(input_video, output_video, start_time_seconds, end_time_seconds)
    grab_frames_at_random(
        input_video=r"C:\Users\yannick.gibson\projects\work\important\ball-tracker\data\videos\blurry\42min.mp4",
        output_folder=r"C:\Users\yannick.gibson\projects\school\BP\bachelors_thesis\annotation\mydata\images\blurry\42min",
        num_frames=2000,
        clean_output_folder=False
    )

    #preprocess_video(
    #    input_path=r"C:\Users\yannick.gibson\projects\work\important\ball-tracker\videos\blurred\2hrs.ts",
    #    output_path=None,
    #    minutes_from=2,
    #    minutes_to=3,
    #    target_fps=24
    #)

