"""This script server for changing a class number in yolo output label txt files.
structure:
    - classes.txt
    - labels
        - 0.txt
        - ...
        - n.txt
    - images
        - 0.jpg
        - ...
        - n.jpg
"""
import os
from tqdm import tqdm

def change_labels(labels_folder: str, class_from: int, class_to: int) -> None:
    """Change class number in all label_{number}.txt files in labels folder."""
    file_names = os.listdir(labels_folder)
    print(f"Renaming class of index {class_from} to a class of index {class_to} in {len(file_names)} files in {labels_folder}.")
    for file in tqdm(file_names):
        if file.endswith(".txt"):
            with open(labels_folder + "/" + file, "r") as f:
                lines = f.readlines()
            with open(labels_folder + "/" + file, "w") as f:
                for line in lines:
                    if line[:2] == f"{class_from} ":  # if class number is at the beginning of the line
                        line = f"{class_to} " + line[2:]
                    f.write(line)
    print("Done.")
def main() -> None:
    parent_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") 
    yolo_folder = parent_folder + "/mydata/yolo_output/blurry/2hrs_30fps_1to2min_0.83x_speed"
    labels_folder = yolo_folder + "/labels"
    class_from = 3  # previously scorekeeper
    class_to = 4  # now scorekeeper

#    # Change classes.txt
#    with open(yolo_folder + "/classes.txt", "w") as f:
#        classes_txt = \
#"""ball
#paddle
#player
#player serving
#scorekeeper"""
#        f.write(classes_txt)

    change_labels(labels_folder, class_from=class_from, class_to=class_to)


if __name__ == "__main__":
    main()