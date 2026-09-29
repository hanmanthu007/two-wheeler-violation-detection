from ultralytics import YOLO
import cv2
import os

# ==========================================
# HELPER: FILTER DUPLICATE RIDER BOXES
# ==========================================

def filter_duplicate_riders(boxes, x_thresh_ratio=0.18, iou_thresh=0.60):
    """Filters duplicate detections of the same rider."""
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda x: x["conf"], reverse=True)
    kept = []

    for b in boxes:
        bx1, by1, bx2, by2 = b["box"]
        b_center_x = (bx1 + bx2) / 2.0
        b_width = bx2 - bx1
        b_area = b_width * (by2 - by1)

        is_duplicate = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k["box"]
            k_center_x = (kx1 + kx2) / 2.0
            k_width = kx2 - kx1

            x_dist = abs(b_center_x - k_center_x)
            avg_width = (b_width + k_width) / 2.0

            ix1, iy1 = max(bx1, kx1), max(by1, ky1)
            ix2, iy2 = min(bx2, kx2), min(by2, ky2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter_area = iw * ih
            k_area = (kx2 - kx1) * (ky2 - ky1)
            union_area = b_area + k_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0

            if (x_dist < avg_width * x_thresh_ratio) or (iou > iou_thresh):
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(b)

    return kept


# ==========================================
# LOAD MODEL & IMAGE
# ==========================================

print("=" * 50)
print("      TWO-WHEELER TRIPLE RIDING DETECTION")
print("=" * 50)

model = YOLO("yolov8n.pt")
image_path = "test.jpg.jpg"

if not os.path.exists(image_path):
    print(f"ERROR: Image not found -> {image_path}")
    exit()

img = cv2.imread(image_path)
if img is None:
    print("ERROR: Unable to read image.")
    exit()


# ==========================================
# RUN DETECTION
# ==========================================

print("Running detection...")
results = model(image_path, conf=0.30, verbose=False)

raw_persons = []
motorcycles = []

for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]
        coords = list(map(int, box.xyxy[0]))

        if label == "person":
            raw_persons.append({"box": coords, "conf": conf})
        elif label == "motorcycle":
            motorcycles.append({"box": coords, "conf": conf})

# Filter out duplicate bounding boxes
persons = filter_duplicate_riders(raw_persons, x_thresh_ratio=0.18, iou_thresh=0.60)
person_count = len(persons)
bike_count = len(motorcycles)


# ==========================================
# DRAW BOUNDING BOXES
# ==========================================

# Draw Motorcycles (Blue)
for m in motorcycles:
    x1, y1, x2, y2 = m["box"]
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img, f"Motorcycle {m['conf']:.2f}", (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

# Draw Riders (Green)
for p in persons:
    x1, y1, x2, y2 = p["box"]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"Rider {p['conf']:.2f}", (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


# ==========================================
# EVALUATE TRIPLE RIDING STATUS
# ==========================================

if bike_count >= 1:
    if person_count == 3:
        status_text = "TRIPLE RIDING DETECTED (3 Riders)"
        status_color = (0, 0, 255)  # Red
    elif person_count > 3:
        status_text = f"OVERLOADED ({person_count} Riders)"
        status_color = (0, 0, 255)  # Red
    else:
        status_text = f"NORMAL RIDING ({person_count} Rider{'s' if person_count > 1 else ''})"
        status_color = (0, 180, 0)  # Green
else:
    status_text = "NO MOTORCYCLE FOUND"
    status_color = (0, 140, 255)  # Orange

# Draw Upper Status Card
cv2.rectangle(img, (15, 15), (500, 85), (255, 255, 255), -1)
cv2.rectangle(img, (15, 15), (500, 85), (60, 60, 60), 2)
cv2.putText(img, "RIDING LOAD STATUS:", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 2)
cv2.putText(img, status_text, (25, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)


# ==========================================
# TERMINAL REPORT & OUTPUT
# ==========================================

print("\n" + "=" * 40)
print("         TRIPLE RIDING REPORT")
print("=" * 40)
print(f"Persons Detected     : {person_count}")
print(f"Motorcycles Detected : {bike_count}")
print(f"Status               : {status_text}")
print("=" * 40)

output_path = "triple_riding_output.jpg"
cv2.imwrite(output_path, img)
print(f"\nOutput saved as      : {output_path}")

cv2.imshow("Triple Riding Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()