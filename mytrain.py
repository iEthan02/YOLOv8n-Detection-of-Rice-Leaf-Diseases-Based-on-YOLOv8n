import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolov8n.pt")

    model.train(
        data=r"I:\deeplearning\ultralytics-8.3.163\datasets\yolo_dataset\data.yaml",
        epochs=300,
        imgsz=640,
        batch=16,
        workers=0,
        device=0,
        amp=True
    )