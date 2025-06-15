import os
from ultralytics import YOLO
from PIL import Image

# ✅ CONFIGURE THESE
model_path = "runs/detect/train5/weights/best.pt"   # Path to your trained YOLOv8 model
input_image_path = "/Users/shreyassharma/Desktop/yolo/inference/images/a.png"           # ✅ Change this to your input image path
output_cropped_dir = "cropped_output"              # Folder where cropped image will be saved
conf_threshold = 0.8                             # Confidence threshold for detection

# ✅ Load the model
model = YOLO(model_path)

# ✅ Create output directory if it doesn't exist
os.makedirs(output_cropped_dir, exist_ok=True)

# ✅ Run detection
results = model(input_image_path, conf=conf_threshold)

# ✅ Extract detections
img = Image.open(input_image_path)
basename = os.path.splitext(os.path.basename(input_image_path))[0]

for i, box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
    cropped = img.crop((x1, y1, x2, y2))

    cropped_path = os.path.join(output_cropped_dir, f"{basename}_crop{i+1}.png")
    cropped.save(cropped_path)
    print(f"✅ Cropped Aadhaar saved to: {cropped_path}")
results[0].save(filename="debug_output.jpg")
