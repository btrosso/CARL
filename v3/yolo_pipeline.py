# Make sure ultralytics is installed first: pip install ultralytics
import os
from ultralytics import YOLO
from ultralytics.utils import LOGGER

def train_yolov8_model(
    model_size='yolov8n.pt',
    data_yaml='/media/ssdset/users/brosso/workspace/CARL/v3/vehicle_axle_dataset/data.yaml',
    epochs=15,  #50
    imgsz=640,
    batch=8,  #16
    project='runs/train',
    name='vehicle_axle_v28'
):
    LOGGER.setLevel("INFO")
    print("🟡 Starting YOLOv8 training on CPU...")
    model = YOLO(model_size)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="0",  # change this if / when i have a gpu
        project=project,
        name=name
    )
    print("✅ Training complete.")
    return results

def evaluate_yolov8_model(
    model_path='/Users/brosso/Documents/personal_code/CARL/old/v3/runs/train/vehicle_axle_v1/weights/best.pt',
    data_yaml='/Users/brosso/Documents/personal_code/CARL/old/v3/vehicle_axle_dataset/data.yaml'
):
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)
    return metrics

def predict_yolov8_model(
    model_path='/Users/brosso/Documents/personal_code/CARL/v3/runs/train/vehicle_axle_v29/weights/best.pt',
    source_dir='/Users/brosso/Documents/personal_code/CARL/algotraffic_low_qual/05282025_pt1',
    # source_dir='/media/ssdset/users/brosso/workspace/CARL/v3/vehicle_axle_dataset/images/val',
    save_dir='runs/predict',
    conf_threshold=0.25  # 👈 Default confidence threshold
):
    model = YOLO(model_path)
    results = model.predict(
        source=source_dir,
        save=True,
        save_txt=True,
        conf=0.70,   # 👈 Adjust threshold here  .25 for accuracy test | .60 for inference / annotation pipeline
        project=save_dir
    )
    return results



if __name__ == "__main__":
    # Train:
    # train_yolov8_model()

    # Evaluate:
    # evaluate_yolov8_model()

    # Predict and save annotated results:
    predict_yolov8_model()
