from ultralytics import YOLO
import cv2
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Swap with yolov8s.pt for better accuracy

# Define class ID for vehicles (based on COCO dataset)
COCO_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 6: "train", 7: "truck"}
vehicle_ids = set(COCO_CLASSES.keys())  # Filter only vehicle-related detections

#################################################################################
# DON'T CHANGE THIS SECTION                                                     #
#################################################################################
# Input and output directories
image_dir = "/Users/brosso/Documents/personal_code/CARL/data2"
output_dir = "/Users/brosso/Documents/personal_code/CARL/annotated_images"
label_dir = "/Users/brosso/Documents/personal_code/CARL/annotation_text_files"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)
#################################################################################

# Process each image
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    img = cv2.imread(image_path)
    
    # Run inference
    results = model(image_path)

    annotation_lines = []
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id in vehicle_ids:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x_center = (x1 + x2) / (2 * img.shape[1])
                y_center = (y1 + y2) / (2 * img.shape[0])
                width = (x2 - x1) / img.shape[1]
                height = (y2 - y1) / img.shape[0]
                
                annotation_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")

        # Save annotated image
        img_with_boxes = result.plot()
        cv2.imwrite(os.path.join(output_dir, image_name), img_with_boxes)

        # Save annotation file
        annotation_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt").replace(".png", ".txt"))
        with open(annotation_path, "w") as f:
            f.write("\n".join(annotation_lines))

print("Processing complete. Check 'output_images' for images and 'annotations' for bounding box labels.")