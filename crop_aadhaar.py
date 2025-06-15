from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load trained model
model = YOLO("runs/detect/train5/weights/best.pt")

# Input and output directories
input_folder = "inference/images"
output_folder = "inference/crops"
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.9

# Function to deskew the Aadhaar card using image moments
def correct_rotation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]))
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # Adjust angle (OpenCV returns strange values sometimes)
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    # Rotate the image to deskew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    corrected = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return corrected

# Run detection and cropping
for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)

    results = model(image)[0]

    for i, box in enumerate(results.boxes):
        conf = float(box.conf)
        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]

        # Rotate crop to horizontal orientation
        rotated_crop = correct_rotation(cropped)

        # Save the rotated cropped Aadhaar card
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_aadhaar_{i}.png")
        cv2.imwrite(output_path, rotated_crop)

        print(f"[✓] Saved corrected Aadhaar card: {output_path}")
