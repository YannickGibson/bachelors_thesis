"""
This code's main purpose is to split YOLO dataset into training, validating, and test sets.

Define arguments, to be --train --val --test to be from 0-1 and sum to 1.
"""
import random
import argparse
import os
import shutil
from tqdm import tqdm


def prepare_folders(output_path: str,
                    dataset_path: str,
                    train_rate: float,
                    val_rate: float,
                    test_rate: float
                    ) -> str:
    """Generate folder structure with simple files for the datasets."""
    # Choose output folder name
    dataset_folder_name = dataset_path.split("/")[-1]
    if test_rate != 0:
        output_folder_name = f"{dataset_folder_name}_train{train_rate}_val{val_rate}_test{test_rate}"
    else:
        output_folder_name = f"{dataset_folder_name}_train{train_rate}_val{val_rate}"

    # Create output folder
    output_folder_path = f"{output_path}/{output_folder_name}"
    if os.path.exists(output_folder_path):
        print(f"Output folder '{output_folder_path}' already exists.")
        while True:
            user_input = input("Wipe folder and regenerate? (y/n): ")
            if user_input.lower() == "y":
                shutil.rmtree(output_folder_path)
                break
            elif user_input.lower() == "n":
                raise FileExistsError("Output folder already exists")

    os.makedirs(output_folder_path, exist_ok=True)

    # Create train, val, (test) folders
    subfolders = ["train", "val", "test"] if test_rate != 0 else ["train", "val"]
    for subfolder in subfolders:
        os.makedirs(f"{output_folder_path}/{subfolder}", exist_ok=True)
        os.makedirs(f"{output_folder_path}/{subfolder}/images", exist_ok=True)
        os.makedirs(f"{output_folder_path}/{subfolder}/labels", exist_ok=True)
        shutil.copyfile(f"{dataset_path}/classes.txt", f"{output_folder_path}/{subfolder}/classes.txt")

    # Create data.yaml for YOLO training
    classes: list[str] = []
    with open(f"{dataset_path}/classes.txt", "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line != "":
                classes.append(line)
    with open(f"{output_folder_path}/data.yaml", "w") as f:
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        if test_rate != 0:
            f.write("test: test/images\n")
        f.write(f"nc: {len(classes)}\n")
        classes_with_quotes = [f"'{c}'" for c in classes]
        f.write(f"classes: [{', '.join(classes_with_quotes)}]\n")

    return output_folder_path


def split_dataset(dataset_path: str,
                  output_path: str,
                  train_rate: float,
                  val_rate: float,
                  test_rate: float
                  ) -> None:
    """Split the dataset into training, validating, and testing sets."""
    
    # Check sum of rates
    if train_rate + val_rate + test_rate != 1:
        raise ValueError("Sum of rates must be equal to 1.")
    
    output_folder_path = prepare_folders(output_path, dataset_path, train_rate, val_rate, test_rate)

    # Get all image files in the dataset (expect labels have same name jpg/txt)
    image_paths = os.listdir(dataset_path + "/images")
    random.shuffle(image_paths)

    # Split the paths
    train_count = int(len(image_paths) * train_rate)
    if test_rate == 0:
        val_count = len(image_paths) - train_count
        test_count = 0
    else:
        val_count = int(len(image_paths) * val_rate)
        test_count = len(image_paths) - train_count - val_count
    
    train_paths: list[str] = image_paths[:train_count]
    val_paths: list[str] = image_paths[train_count:train_count + val_count]
    test_paths: list[str] = image_paths[train_count + val_count:train_count + val_count + test_count]

    # Copy images and labels to the output folder
    data = {"train": train_paths, "val": val_paths, "test": test_paths}
    for subfolder, paths in data.items():
        if len(paths) == 0:
            continue
        print(f"Copying {len(paths)} images ({len(paths)/len(image_paths)}) and labels to {subfolder}.")
        for path in tqdm(paths):
            image_name = ".".join(path.split(".")[:-1])  # last part is extension
            ext = path.split(".")[-1]
            shutil.copyfile(f"{dataset_path}/images/{image_name}.{ext}", f"{output_folder_path}/{subfolder}/images/{image_name}.{ext}")
            shutil.copyfile(f"{dataset_path}/labels/{image_name}.txt", f"{output_folder_path}/{subfolder}/labels/{image_name}.txt")
    print("Done.")


def main() -> None:
    # Define arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_path", type=str, required=True)
    argparser.add_argument("--output_path", type=str, default="data/yolo_datasets")
    argparser.add_argument("--train", type=float, default=0.75)
    argparser.add_argument("--val", type=float, default=0.25)
    argparser.add_argument("--test", type=float, default=0.0)
    args = argparser.parse_args()

    # Extract arguments
    train_rate = args.train
    val_rate = args.val
    test_rate = args.test
    dataset_path = os.path.abspath(args.dataset_path.replace("\\", "/")).replace("\\", "/")
    output_path = os.path.abspath(args.output_path.replace("\\", "/")).replace("\\", "/")

    # Split Dataset
    split_dataset(dataset_path, output_path, train_rate, val_rate, test_rate)


if __name__ == "__main__":
    main()
