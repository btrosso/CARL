import os
from pathlib import Path
import numpy as np

# Threshold for IoU to count a match
IOU_THRESHOLD = 0.5

def load_yolo_txt(file_path):
    if not file_path.exists():
        return np.zeros((0, 5))  # class_id + bbox
    with open(file_path) as f:
        lines = [list(map(float, line.strip().split())) for line in f]
    return np.array(lines)

def compute_iou(boxA, boxB):
    # YOLO format: [x_center, y_center, width, height]
    xa1 = boxA[0] - boxA[2] / 2
    ya1 = boxA[1] - boxA[3] / 2
    xa2 = boxA[0] + boxA[2] / 2
    ya2 = boxA[1] + boxA[3] / 2

    xb1 = boxB[0] - boxB[2] / 2
    yb1 = boxB[1] - boxB[3] / 2
    xb2 = boxB[0] + boxB[2] / 2
    yb2 = boxB[1] + boxB[3] / 2

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    boxA_area = (xa2 - xa1) * (ya2 - ya1)
    boxB_area = (xb2 - xb1) * (yb2 - yb1)

    iou = inter_area / (boxA_area + boxB_area - inter_area + 1e-6)
    return iou

def match_predictions(gt_boxes, pred_boxes):
    matched = 0
    used_pred = set()
    for gt in gt_boxes:
        for i, pred in enumerate(pred_boxes):
            if i in used_pred:
                continue
            if int(gt[0]) == int(pred[0]) and compute_iou(gt[1:], pred[1:]) >= IOU_THRESHOLD:
                matched += 1
                used_pred.add(i)
                break
    return matched

def evaluate_yolo_accuracy(gt_dir, pred_dir):
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    gt_files = list(gt_dir.glob("*.txt"))

    total_gt = 0
    total_matched = 0

    for gt_file in gt_files:
        pred_file = pred_dir / gt_file.name
        gt_boxes = load_yolo_txt(gt_file)
        pred_boxes = load_yolo_txt(pred_file)

        total_gt += len(gt_boxes)
        total_matched += match_predictions(gt_boxes, pred_boxes)

    accuracy = total_matched / total_gt if total_gt > 0 else 0
    print(f"✅ Detection Accuracy: {accuracy*100:.2f}%")
    print(f"📦 Matched: {total_matched} / {total_gt} ground-truth boxes")
"""
RESULTS:
v16 ~ 80.67%
v17 ~ 83.52%
v18 ~ 79.72%
v19 ~ 78.10%
v20 ~ 83.66%
v21 ~ 82.45%
v22 ~ 83.79%
v23 ~ 75.69%
"""

if __name__ == "__main__":
    evaluate_yolo_accuracy(
        gt_dir="/Users/brosso/Documents/personal_code/CARL/v3/vehicle_axle_dataset/labels/val",
        pred_dir="/Users/brosso/Documents/personal_code/CARL/v3/runs/predict/predict/labels"  # adjust if different
    )
