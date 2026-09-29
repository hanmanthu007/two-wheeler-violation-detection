from ultralytics import YOLO
import cv2
import os

# =========================================================
# 1. LOAD MODEL & IMAGE
# =========================================================

print("Loading Helmet Detection Model...")
model_path = "models/helmet_model.pt"
image_path = "test.jpg.jpg"

if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    exit()

if not os.path.exists(image_path):
    print(f"ERROR: Image not found at {image_path}")
    exit()

model = YOLO(model_path)
img = cv2.imread(image_path)

# =========================================================
# 2. RUN INFERENCE & COUNT
# =========================================================

print("Running helmet detection...\n")
results = model(image_path, conf=0.30, verbose=False)

with_helmet_count = 0
without_helmet_count = 0

for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label_lower = label.lower().strip()

        # Robust substring detection
        is_without_helmet = "without" in label_lower or "no" in label_lower

        if is_without_helmet:
            without_helmet_count += 1
            color = (0, 0, 255)  # Red
            tag = f"NO HELMET {conf:.2f}"
        else:
            with_helmet_count += 1
            color = (0, 255, 0)  # Green
            tag = f"HELMET {conf:.2f}"

        print(f"Detected: {label:<20} | Confidence: {conf:.2f}")

        # Draw bounding box and label tag
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            tag,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

# =========================================================
# 3. TERMINAL REPORT
# =========================================================

print("\n" + "=" * 40)
print("             HELMET REPORT")
print("=" * 40)
print(f"With Helmet Detected     : {with_helmet_count}")
print(f"Without Helmet Detected  : {without_helmet_count}")
print("-" * 40)

if without_helmet_count > 0:
    print("STATUS: [!] HELMET VIOLATION DETECTED")
else:
    print("STATUS: [OK] NO HELMET VIOLATION")

print("=" * 40)

# =========================================================
# 4. PREVIEW WINDOW
# =========================================================

cv2.imshow("Helmet Detection Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()