from ultralytics import YOLO
import cv2
import os

# ==========================================
# HELPER: REMOVE DUPLICATE RIDER BOXES
# ==========================================

def filter_duplicate_riders(boxes, x_thresh_ratio=0.18, iou_thresh=0.60):
    """Suppresses duplicate bounding boxes for the same individual."""
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
# 1. LOAD MODEL & RUN DETECTION
# ==========================================

print("=" * 55)
print("      TWO-WHEELER RIDER-TO-BIKE ASSOCIATION")
print("=" * 55)

model = YOLO("yolov8n.pt")
image_path = "test.jpg.jpg"

if not os.path.exists(image_path):
    print(f"ERROR: Image not found -> {image_path}")
    exit()

img = cv2.imread(image_path)
results = model(image_path, conf=0.25, verbose=False)

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
            motorcycles.append({"box": coords, "conf": conf, "riders": []})

# Clean duplicate detections on seated riders
persons = filter_duplicate_riders(raw_persons, x_thresh_ratio=0.18, iou_thresh=0.60)


# ==========================================
# 2. ASSOCIATE RIDERS TO MOTORCYCLES
# ==========================================

for person in persons:
    px1, py1, px2, py2 = person["box"]
    person_center_x = (px1 + px2) / 2.0
    person_bottom_y = py2

    matched_bike = None
    max_overlap = 0

    for bike in motorcycles:
        bx1, by1, bx2, by2 = bike["box"]

        # Horizontal alignment: Person center is within bike width (with 10% margin)
        bike_width = bx2 - bx1
        is_x_aligned = (bx1 - 0.10 * bike_width) <= person_center_x <= (bx2 + 0.10 * bike_width)

        # Vertical alignment: Person's lower body sits on or overlaps the motorcycle
        is_y_aligned = (person_bottom_y >= by1) and (py1 <= by2)

        if is_x_aligned and is_y_aligned:
            # Measure horizontal overlap as match score
            overlap_x = max(0, min(px2, bx2) - max(px1, bx1))
            if overlap_x > max_overlap:
                max_overlap = overlap_x
                matched_bike = bike

    if matched_bike is not None:
        matched_bike["riders"].append(person)


# ==========================================
# 3. DRAW ANNOTATIONS & DECIDE VIOLATIONS
# ==========================================

total_associated_riders = 0
has_triple_riding = False

# Draw Motorcycles (Blue)
for i, bike in enumerate(motorcycles, start=1):
    bx1, by1, bx2, by2 = bike["box"]
    rider_count = len(bike["riders"])
    total_associated_riders += rider_count

    cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
    cv2.putText(
        img,
        f"Motorcycle #{i} | Riders: {rider_count}",
        (bx1, max(by1 - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 0, 0),
        2
    )

    if rider_count >= 3:
        has_triple_riding = True

# Draw Riders (Green)
for p in persons:
    x1, y1, x2, y2 = p["box"]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img,
        f"Rider {p['conf']:.2f}",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

# Status Overlay Card
cv2.rectangle(img, (15, 15), (460, 90), (255, 255, 255), -1)
cv2.rectangle(img, (15, 15), (460, 90), (60, 60, 60), 2)

if has_triple_riding:
    status_text = f"TRIPLE RIDING VIOLATION ({total_associated_riders} Riders)"
    status_color = (0, 0, 255)
elif len(motorcycles) > 0:
    status_text = f"NORMAL RIDING ({total_associated_riders} Riders)"
    status_color = (0, 180, 0)
else:
    status_text = "NO MOTORCYCLE DETECTED"
    status_color = (0, 140, 255)

cv2.putText(img, "RIDER LOAD STATUS:", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
cv2.putText(img, status_text, (25, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)


# ==========================================
# 4. TERMINAL SUMMARY & SAVE
# ==========================================

print("\n" + "=" * 40)
print("       DETECTION SUMMARY REPORT")
print("=" * 40)
print(f"Total Persons Detected     : {len(persons)}")
print(f"Total Motorcycles Detected : {len(motorcycles)}")
print(f"Total Associated Riders    : {total_associated_riders}")
print("-" * 40)

for idx, bike in enumerate(motorcycles, start=1):
    print(f"Motorcycle {idx} Riders        : {len(bike['riders'])}")

print(f"Final Status               : {status_text}")
print("=" * 40)

output_path = "associated_riders_output.jpg"
cv2.imwrite(output_path, img)
print(f"Output image saved to      : {output_path}")

cv2.imshow("Rider-Motorcycle Association", img)
cv2.waitKey(0)
cv2.destroyAllWindows()