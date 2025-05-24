import os
import csv
import shutil

CSV_FILE = "dataset_record.csv"
SOURCE_DIR = "/path/to/flat_dir"  # Contains /images and /labels
TARGET_ROOT = "/path/to/reconstructed_vehicle_axle_dataset"

def reconstruct_from_csv(csv_file, source_dir, target_root):
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            rel_path = row['file_path']
            split = row['split']
            file_type = row['type']

            if file_type == "image":
                subdir = "images"
            elif file_type == "label":
                subdir = "labels"
            else:
                continue  # skip unknown types

            # Define paths
            filename = os.path.basename(rel_path)
            src_file = os.path.join(source_dir, subdir, filename)
            dest_dir = os.path.join(target_root, subdir, split)
            dest_file = os.path.join(dest_dir, filename)

            # Create destination directory if it doesn't exist
            os.makedirs(dest_dir, exist_ok=True)

            # Move the file
            if os.path.exists(src_file):
                shutil.move(src_file, dest_file)
                print(f"Moved {src_file} → {dest_file}")
            else:
                print(f"⚠️  File not found: {src_file}")

if __name__ == "__main__":
    reconstruct_from_csv(CSV_FILE, SOURCE_DIR, TARGET_ROOT)
