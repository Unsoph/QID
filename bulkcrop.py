from ultralytics import YOLO
from PIL import Image
import os
from glob import glob

# Load the trained YOLOv8 model
model = YOLO("runs/detect/train2/weights/best.pt")

# Folder containing input images
input_folder = "/Users/shreyassharma/Desktop/yolo/images"  # 🔁 Replace this with your folder path

# Output folder for cropped images
output_folder = "cropped_results_batch"
os.makedirs(output_folder, exist_ok=True)

# Supported image extensions
image_extensions = ['*.jpg', '*.jpeg', '*.png']

# Get all image file paths
image_files = []
for ext in image_extensions:
    image_files.extend(glob(os.path.join(input_folder, ext)))

print(f"Found {len(image_files)} images in '{input_folder}'")

# Process each image
for img_path in image_files:
    # Run inference
    results = model(img_path)[0]
    
    # Open image
    image = Image.open(img_path)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # Crop and save all detected Aadhaar cards in this image
    for i, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box.tolist())
        cropped = image.crop((x1, y1, x2, y2))
        save_path = os.path.join(output_folder, f"{base_name}_crop_{i+1}.jpg")
        cropped.save(save_path)
    
    print(f"Processed '{img_path}': cropped {len(results.boxes)} Aadhaar card(s)")

print(f"All crops saved to '{output_folder}'")