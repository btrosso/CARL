import os
from pathlib import Path
import shutil

def isolate_high_class_predictions(
    pred_labels_dir,
    source_images_dir,
    output_labels_dir,
    output_images_dir,
    class_threshold=4
):
    pred_labels_dir = Path(pred_labels_dir)
    source_images_dir = Path(source_images_dir)
    output_labels_dir = Path(output_labels_dir)
    output_images_dir = Path(output_images_dir)

    output_labels_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    moved_count = 0

    for label_file in pred_labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            class_id = int(line.strip().split()[0])
            if class_id > class_threshold:
                # Move label
                dest_label_path = output_labels_dir / label_file.name
                shutil.move(str(label_file), str(dest_label_path))

                # Copy corresponding image
                image_name = label_file.with_suffix(".jpg").name  # assumes .jpg
                src_img_path = source_images_dir / image_name
                if src_img_path.exists():
                    dest_img_path = output_images_dir / image_name
                    shutil.copy(src_img_path, dest_img_path)
                    moved_count += 1
                break  # No need to check more lines in this file

    print(f"✅ Done. Moved {moved_count} images and their label files.")

if __name__ == "__main__":
    isolate_high_class_predictions(
        pred_labels_dir="/Users/brosso/Documents/personal_code/CARL/v3/runs/predict/predict/labels",
        source_images_dir="/Users/brosso/Documents/personal_code/CARL/algotraffic_low_qual/05142025",
        output_labels_dir="/Users/brosso/Documents/personal_code/CARL/v3/runs/ISOLATE/labels",
        output_images_dir="/Users/brosso/Documents/personal_code/CARL/v3/runs/ISOLATE/images"
    )
