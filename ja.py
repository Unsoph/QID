import torch
from ultralytics import YOLO

# ✅ Detect device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training will run on: {device}")
print("MPS Available:", torch.backends.mps.is_available())
print("MPS Built:", torch.backends.mps.is_built())

# ✅ Train YOLOv8
model = YOLO("yolov8n.pt")
model.train(data="my-first-project-4/data.yaml", epochs=60, imgsz=640, device=device)