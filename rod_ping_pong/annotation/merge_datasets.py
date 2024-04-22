from shutil import copyfile
import os


def merge(*dataset_paths, output_folder) -> None:
    """Merge X amount of YOLO format datasets into one."""
    if len(dataset_paths) < 2:
        raise ValueError("You need to provide at least 2 datasets to merge.")
    
    # Make dataset_paths unix compatible
    dataset_paths = [path.replace("\\", "/") for path in dataset_paths]

    # get dataset names
    dataset_names = [path.split("/")[-1] for path in dataset_paths]
    output_name = "__".join(dataset_names)
    output_path = os.path.join(output_folder, output_name)

    # Create the output folder
    os.makedirs(output_path, exist_ok=True)

    # Check if classes.txt in all datasets are the same
    classes = []
    for path in dataset_paths:
        with open(path + "/classes.txt", "r") as f:
            classes.append(f.read())

    if len(set(classes)) != 1:
        print(classes)
        raise ValueError("All classes.txt files must be the same.")    

    # Copy classes.txt based on the first dataset
    copyfile(dataset_paths[0] + "/classes.txt", output_path + "/classes.txt")
    return
    #copy_all files from one folder to another
    for i in range(len(dataset_paths)):
        for file in os.listdir(dataset_paths[i]):
            copyfile(dataset_paths[i]+"/"+file, output_folder+"/"+file)


def main() -> None:
    parent_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

    DATASET_PATH_1 = parent_folder + "/mydata/yolo_output/blurry/2hrs_30fps_1to2min_0.83x_speed"
    DATASET_PATH_2 = parent_folder + "/mydata/yolo_output/blurry/42min_100random"
    DATASET_PATH_3 = parent_folder + "/mydata/yolo_output/blurry/42min_1000random"

    OUTPUT_DATASET_FOLDER_PATH = "mydata/yolo_output/blurry/"

    merge(DATASET_PATH_1, DATASET_PATH_2, DATASET_PATH_3, output_folder=OUTPUT_DATASET_FOLDER_PATH)
    
    pass


if __name__ == "__main__":
    main()