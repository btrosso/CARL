import os

annotation_files = os.listdir("/Users/brosso/Documents/personal_code/CARL/yolo_annotations")
print(f"Num Annotation Files: {len(annotation_files)}")

image_files = os.listdir("/Users/brosso/Documents/personal_code/CARL/yt_traffic_high_quality")
print(f"Num Image Files: {len(image_files)}")