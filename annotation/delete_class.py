"""This script server for deleting certain classes from a dataset. 
Note: You have to adjust data.yaml/classes.txt manually, script only changes the labels/ folder
Warning: This operation cannot be undone! Make sure to backup your data before running this script.

YOLO Dataset structure:
    - dataset
        - classes.txt / data.yaml
        - labels
            - frame_0.txt
            - ...
            - frame_n.txt
        - images
            - frame_0.jpg
            - ...
            - frame_n.jpg
"""


import os
from tqdm import tqdm


def change_labels(labels_folder: str, delete_indexes: list[int]) -> None:
    """Change class number in all label_{number}.txt files in labels folder."""
    file_names = os.listdir(labels_folder)
    print(f"Deleting classes {delete_indexes} in {labels_folder}.")
    for file in tqdm(file_names):
        if file.endswith(".txt"):
            # Read lines
            with open(labels_folder + "/" + file, "r") as f:
                lines = f.readlines()
            # Write new lines
            with open(labels_folder + "/" + file, "w") as f:
                for line in lines:
                    # write only classes other then delete_indexes
                    if int(line[:1]) not in delete_indexes: 
                        f.write(line)

    print("Done.")



def main() -> None:
    RUN = True
    if RUN:
        yolo_folder = r"C:\Users\yannick.gibson\projects\work\experiments\yolov7\data\42min_1000random_ball_only"

        labels_folder = yolo_folder + "/labels"
        change_labels(labels_folder, delete_indexes=[1, 2, 3, 4])
    else:
        print("Set RUN = True to run the script.")


if __name__ == "__main__":
    main()
