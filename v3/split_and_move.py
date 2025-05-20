import os
import shutil
import random
from pathlib import Path

def split_yolo_dataset(images_dir, annotations_dir, output_dir, val_split=0.2, seed=42):
    random.seed(seed)

    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)
    output_dir = Path(output_dir)

    # Ensure output structure
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Find all matching annotation/image pairs
    matched_pairs = []
    for label_file in annotations_dir.glob("*.txt"):
        image_stem = label_file.stem
        for ext in ['.jpg', '.jpeg', '.png']:
            image_path = images_dir / f"{image_stem}{ext}"
            if image_path.exists():
                matched_pairs.append((image_path, label_file))
                break

    # Shuffle and split
    random.shuffle(matched_pairs)
    split_index = int(len(matched_pairs) * (1 - val_split))
    train_pairs = matched_pairs[:split_index]
    val_pairs = matched_pairs[split_index:]

    # Move files
    for split_name, pairs in [('train', train_pairs), ('val', val_pairs)]:
        for img_path, lbl_path in pairs:
            shutil.move(str(img_path), output_dir / 'images' / split_name / img_path.name)
            shutil.move(str(lbl_path), output_dir / 'labels' / split_name / lbl_path.name)

    return {
        'train_count': len(train_pairs),
        'val_count': len(val_pairs),
        'total_matched': len(matched_pairs)
    }

if __name__ == "__main__":
    result = split_yolo_dataset(
        "/Users/brosso/Documents/personal_code/CARL/algotraffic_low_qual/05152025", 
        "/Users/brosso/Documents/personal_code/CARL/yolo_annotations", 
        "vehicle_axle_dataset"
        )
    print(result)
