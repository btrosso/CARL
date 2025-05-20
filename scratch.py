import os

annotation_files = os.listdir("/Users/brosso/Documents/personal_code/CARL/yolo_annotations")
print(f"Num Annotation Files: {len(annotation_files)}")

image_files = os.listdir("/Users/brosso/Documents/personal_code/CARL/yt_traffic_high_quality/yt1_r2")
print(f"Num YT1_R2 Image Files: {len(image_files)}")

train_files = os.listdir("/Users/brosso/Documents/personal_code/CARL/v3/vehicle_axle_dataset/images/train")
print(f"Num Train Files: {len(train_files)}")

val_files = os.listdir("/Users/brosso/Documents/personal_code/CARL/v3/vehicle_axle_dataset/images/val")
print(f"Num Validation Files: {len(val_files)}")