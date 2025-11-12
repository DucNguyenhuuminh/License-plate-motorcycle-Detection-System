import os
import cv2
from ultralytics import YOLO

# === CONFIGURATION ===
model_path = "runs/detect/lpd_yolo/weights/best.pt"  
input_folder = "data/license_plate/images/test"                         
output_folder = "detection_results"                  

# Create output folder 
os.makedirs(output_folder, exist_ok=True)

# Load YOLO model
model = YOLO(model_path)

# Get image
image_files = [f for f in os.listdir(input_folder)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print(f"🔍 Found {len(image_files)} test images in '{input_folder}'")

for img_name in image_files:
    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠️ Skipping invalid image: {img_path}")
        continue

    results = model(img)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img, (x1, y1 - t_size[1] - 6),
                          (x1 + t_size[0], y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Save result
    save_path = os.path.join(output_folder, img_name)
    cv2.imwrite(save_path, img)

