import cv2
import os
from pathlib import Path
import numpy as np

# === CONFIG ===
val_images_dir = Path("vehicle_axle_dataset/images/val")
val_labels_dir = Path("vehicle_axle_dataset/labels/val")
pred_labels_dir = Path("runs/predict/predict/labels")  # Adjust to your actual pred output
class_colors = {"matched": (0, 255, 0), "unmatched": (0, 0, 255)}  # green/red

IOU_THRESHOLD = 0.5

def load_yolo_labels(label_path):
    if not label_path.exists():
        return np.zeros((0, 5))
    with open(label_path, "r") as f:
        lines = [list(map(float, line.strip().split())) for line in f]
    return np.array(lines)

def compute_iou(boxA, boxB):
    xa1, ya1, xa2, ya2 = xyxy(boxA)
    xb1, yb1, xb2, yb2 = xyxy(boxB)
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    areaA = (xa2 - xa1) * (ya2 - ya1)
    areaB = (xb2 - xb1) * (yb2 - yb1)
    return inter_area / (areaA + areaB - inter_area + 1e-6)

def xyxy(box):
    x, y, w, h = box
    return x - w/2, y - h/2, x + w/2, y + h/2

def draw_boxes(img, gt_boxes, pred_boxes, img_shape, show_labels=True):
    matched_preds = set()
    for i, gt in enumerate(gt_boxes):
        gt_cls, *gt_box = gt
        match_found = False
        for j, pred in enumerate(pred_boxes):
            if j in matched_preds:
                continue
            pred_cls, *pred_box = pred
            if int(gt_cls) == int(pred_cls) and compute_iou(gt_box, pred_box) >= IOU_THRESHOLD:
                match_found = True
                matched_preds.add(j)
                break

        color = class_colors["matched"] if match_found else class_colors["unmatched"]
        x1, y1, x2, y2 = xyxy(gt_box)
        x1 = int(x1 * img_shape[0])
        y1 = int(y1 * img_shape[1])
        x2 = int(x2 * img_shape[0])
        y2 = int(y2 * img_shape[1])
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Optionally draw class label
        if show_labels:
            label = str(int(gt_cls)+1)  # or use a class name dict here  (+1 here because the class names are just the class ints which are one off from the indexes)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img


def show_validation_results():
    for image_path in sorted(val_images_dir.glob("*.jpg")):
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        name_stem = image_path.stem

        gt_path = val_labels_dir / f"{name_stem}.txt"
        pred_path = pred_labels_dir / f"{name_stem}.txt"

        gt_boxes = load_yolo_labels(gt_path)
        pred_boxes = load_yolo_labels(pred_path)

        img = draw_boxes(img, gt_boxes, pred_boxes, (w, h))

        cv2.imshow("Validation Image (ESC to exit)", img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC key
            break
    cv2.destroyAllWindows()

# Run it
show_validation_results()
