# Entrainement de notre model TOLOv8 sur notre dataset

from ultralytics import YOLO
DATA_YAML = r"data_cars_local.yaml"  # chemin vers data.yaml

model = YOLO("yolov8n.pt")     
model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    batch=16,
    project="runs",
    name="cars_all",
    patience=50
)
