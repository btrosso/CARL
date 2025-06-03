import os

# annotation_files = os.listdir("/media/ssdset/users/brosso/workspace/CARL/yolo_annotations")
# print(f"Num Annotation Files: {len(annotation_files)}")

# image_files = os.listdir("/media/ssdset/users/brosso/workspace/CARL/yt_traffic_high_quality/yt1_r2")
# print(f"Num YT1_R2 Image Files: {len(image_files)}")

train_files = os.listdir("/Users/brosso/Documents/personal_code/CARL/v3/vehicle_axle_dataset/images/train")
print(f"Num Train Files: {len(train_files)}")

val_files = os.listdir("/Users/brosso/Documents/personal_code/CARL/v3/vehicle_axle_dataset/images/val")
print(f"Num Validation Files: {len(val_files)}")

print("----\n")

# algo_imgs2 = os.listdir("/media/ssdset/users/brosso/workspace/CARL/algotraffic_low_qual/05142025")
# print(f"Num Algo Imgs 05142025 Files: {len(algo_imgs2)}")

print("----\n")

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    return total_size

folder_path = "/Users/brosso/Documents/personal_code/CARL/v3/vehicle_axle_dataset/images/train" # Replace with the actual path to the folder
size_in_bytes = get_folder_size(folder_path)

# Convert to more readable units (optional)
size_in_kb = size_in_bytes / 1024
size_in_mb = size_in_kb / 1024
size_in_gb = size_in_mb / 1024

print(f"Folder size: {size_in_bytes} bytes")
print(f"Folder size: {size_in_kb:.2f} KB")
print(f"Folder size: {size_in_mb:.2f} MB")
print(f"Folder size: {size_in_gb:.2f} GB")