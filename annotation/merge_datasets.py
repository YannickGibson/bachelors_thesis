"""Merges arbitraty number of labeled object detection datasets in YOLO format into one.

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

from shutil import copyfile
import os


def merge(*dataset_paths, output_folder) -> None:
    """Merge X amount of YOLO format datasets into one.
    
    Args:
        dataset_paths (str): Paths to the datasets to merge.
        output_folder (str): Path to a folder where a new output folder will be created.
    """
    if len(dataset_paths) < 2:
        raise ValueError("You need to provide at least 2 datasets to merge.")
    
    # Make dataset_paths unix compatible
    dataset_paths = [path.replace("\\", "/") for path in dataset_paths]

    # Check if classes.txt in all datasets are the same
    classes = []
    for path in dataset_paths:
        with open(path + "/classes.txt", "r") as f:
            classes.append(f.read())  # read everything as a string
    if len(set(classes)) != 1:
        print(classes)
        raise ValueError(f"All classes.txt files must be the same. The order of the classes must be the same.: {classes}")
    
    # Create the output folder with its name
    dataset_names = [path.split("/")[-1] for path in dataset_paths]  # Get dataset names
    output_name = "__".join(dataset_names) + "___merged"
    output_path = os.path.join(output_folder, output_name)
    os.makedirs(output_path, exist_ok=True), output_path
    os.makedirs(output_path + "/images", exist_ok=True)
    os.makedirs(output_path + "/labels", exist_ok=True)
    copyfile(dataset_paths[0] + "/classes.txt", output_path + "/classes.txt")  # copy classes.txt from first dataset
    
    # Copy files from dataset's folders specified
    file_names = []
    folders_to_merge = ["images", "labels"]
    for folder in folders_to_merge:
        for i in range(len(dataset_paths)):
            for image_name in os.listdir(dataset_paths[i] + f"/{folder}"):
                # Keep name unique
                file_extension = image_name.split(".")[-1]
                conflict_count = 0
                new_image_name = image_name
                while new_image_name in file_names:
                    conflict_count += 1
                    new_image_name = image_name.split(".")[0] + f"_{conflict_count}." + file_extension
                    print(f"Conflict: {new_image_name}")
                file_names.append(new_image_name)

                # Copy the file
                copy_from = dataset_paths[i] + f"/{folder}/" + image_name
                copy_to = output_path + f"/{folder}/" + new_image_name
                copyfile(copy_from, copy_to)



def main() -> None:
    PARENT_FOLDER = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    DATASET_PATH_1 = PARENT_FOLDER + "/mydata/yolo_output/blurry/2hrs_30fps_1to2min_0.83x_speed"
    DATASET_PATH_2 = PARENT_FOLDER + "/mydata/yolo_output/blurry/42min_100random"
    DATASET_PATH_3 = PARENT_FOLDER + "/mydata/yolo_output/blurry/42min_1000random"

    OUTPUT_DATASET_FOLDER_PATH = PARENT_FOLDER + "/mydata/yolo_output/blurry/"

    merge(DATASET_PATH_1, DATASET_PATH_2, DATASET_PATH_3, output_folder=OUTPUT_DATASET_FOLDER_PATH)


if __name__ == "__main__":
    main()
