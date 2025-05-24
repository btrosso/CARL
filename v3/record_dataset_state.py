import os
import csv
# testing

DATASET_ROOT = "/Users/brosso/Documents/personal_code/CARL/v3/vehicle_axle_dataset"
OUTPUT_CSV = "dataset_record.csv"

def determine_split(path):
    if "/train/" in path:
        return "train"
    elif "/val/" in path:
        return "val"
    else:
        return "unknown"

def process_dataset(dataset_root, output_csv):
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file_path", "split", "type", "label_contents"])
        
        for dirpath, _, filenames in os.walk(dataset_root):
            # print(dirpath)
            for filename in filenames:
                # print(filename)
                if filename == "data.yaml":
                    continue

                full_path = os.path.join(dirpath, filename)
                # print(full_path)
                rel_path = os.path.relpath(full_path, dataset_root)
                # print(rel_path)
                split = determine_split(rel_path)
                # print(split)

                if "images/" in rel_path:
                    file_type = "image"
                    label_contents = ""
                elif "labels/" in rel_path:
                    file_type = "label"
                    with open(full_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                        label_contents = "; ".join([line.strip() for line in f.readlines()])
                        print(label_contents)
                else:
                    continue  # ignore other files

                writer.writerow([rel_path, split, file_type, label_contents])

if __name__ == "__main__":
    process_dataset(DATASET_ROOT, OUTPUT_CSV)
