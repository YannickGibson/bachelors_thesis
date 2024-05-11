"""This script server for changing a class number in yolo output label txt files.

YOLO Dataset structure:
    - dataset
        - classes.txt
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


def change_labels(labels_folder: str, class_from: int, class_to: int) -> None:
    """Change class number in all label_{number}.txt files in labels folder."""
    file_names = os.listdir(labels_folder)
    print(f"Renaming class of index {class_from} to a class of index {class_to} in {len(file_names)} files in {labels_folder}.")
    for file in tqdm(file_names):
        if file.endswith(".txt"):
            # Read lines
            with open(labels_folder + "/" + file, "r") as f:
                lines = f.readlines()
            # Write new lines
            with open(labels_folder + "/" + file, "w") as f:
                for line in lines:
                    # If class matches 'class_from' change it
                    if line[:2] == f"{class_from} ": 
                        line = f"{class_to} " + line[2:]

                    f.write(line)
    print("Done.")


def set_classes_txt(classes_txt_path: str, classes: list[str]) -> None:
    """Change classes.txt file."""
    with open(classes_txt_path, "w") as f:
        classes_txt = "\n".join(classes)
        f.write(classes_txt)


def main() -> None:
    RUN = False
    if RUN:
        parent_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") 
        yolo_folder = parent_folder + "/mydata/yolo_output/blurry/2hrs_30fps_1to2min_0.83x_speed"
        
        # Set classes.txt
        set_classes_txt(yolo_folder + "/classes.txt", ["ball", "paddle", "player", "player serving", "scorekeeper"])

        # Change class number in label.txt files
        labels_folder = yolo_folder + "/labels"
        class_from = 3  # previously scorekeeper
        class_to = 4  # now scorekeeper
        change_labels(labels_folder, class_from=class_from, class_to=class_to)


if __name__ == "__main__":
    main()
