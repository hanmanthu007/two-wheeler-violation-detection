from ultralytics import YOLO
import cv2
import os

# Import OCR and e-Challan modules
from plate_ocr import extract_plate_number
from fine_generator import calculate_and_generate_challan


# =========================================================
# HELPER FUNCTION 1: DRAW BADGE LABEL
# =========================================================

def draw_label(
    image,
    text,
    x,
    y,
    bg_color,
    text_color=(255, 255, 255),
    scale=0.5,
    thickness=1
):
    """Draw a text label with a filled background rectangle."""
    (w, h), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness
    )
    cv2.rectangle(
        image,
        (x, max(0, y - h - 6)),
        (x + w + 6, y + baseline),
        bg_color,
        -1
    )
    cv2.putText(
        image,
        text,
        (x + 3, y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )


# =========================================================
# HELPER FUNCTION 2: REMOVE DUPLICATE RIDER BOXES
# =========================================================

def filter_duplicate_riders(
    boxes,
    x_thresh_ratio=0.18,
    iou_thresh=0.60
):
    """
    Suppresses duplicate bounding boxes for closely seated riders.
    Keeps the detection with highest confidence score.
    """
    if not boxes:
        return []

    boxes = sorted(
        boxes,
        key=lambda x: x["conf"],
        reverse=True
    )

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

            ix1 = max(bx1, kx1)
            iy1 = max(by1, ky1)
            ix2 = min(bx2, kx2)
            iy2 = min(by2, ky2)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
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


# =========================================================
# MAIN PROGRAM
# =========================================================

print("=" * 60)
print("     UNIFIED TWO-WHEELER VIOLATION DETECTION SYSTEM")
print("=" * 60)


# =========================================================
# 1. CHECK REQUIRED FILES
# =========================================================

vehicle_model_path = "yolov8n.pt"
helmet_model_path = "models/helmet_model.pt"

# Detect which image file is present in directory
image_path = "test.jpg.jpg"
if not os.path.exists(image_path):
    for candidate in ["test.jpg.jpg", "test.jpg"]:
        if os.path.exists(candidate):
            image_path = candidate
            break

print("\nChecking required files...")

if not os.path.exists(vehicle_model_path):
    print(f"ERROR: Vehicle model not found -> {vehicle_model_path}")
    exit()

if not os.path.exists(helmet_model_path):
    print(f"ERROR: Helmet model not found -> {helmet_model_path}")
    exit()

if not os.path.exists(image_path):
    print(f"ERROR: Input image not found -> {image_path}")
    exit()


# =========================================================
# 2. LOAD MODELS
# =========================================================

print("\nLoading vehicle/person model...")
vehicle_model = YOLO(vehicle_model_path)
print("Vehicle model loaded.")

print("\nLoading helmet detection model...")
helmet_model = YOLO(helmet_model_path)
print("Helmet model loaded.")


# =========================================================
# 3. READ IMAGE
# =========================================================

img = cv2.imread(image_path)
if img is None:
    print(f"\nERROR: Unable to read input image -> {image_path}")
    exit()

print(f"\nInput image loaded successfully: {image_path}")


# =========================================================
# 4. VEHICLE + PERSON DETECTION
# =========================================================

print("\n[1/4] Running vehicle and rider detection...")

vehicle_results = vehicle_model(
    image_path,
    conf=0.25,
    verbose=False
)

raw_persons = []
motorcycles = []

for result in vehicle_results:
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = vehicle_model.names[cls]
        coords = list(map(int, box.xyxy[0]))

        if label == "person":
            raw_persons.append({
                "box": coords,
                "conf": conf
            })
        elif label == "motorcycle":
            motorcycles.append({
                "box": coords,
                "conf": conf,
                "riders": []
            })

persons = filter_duplicate_riders(
    raw_persons,
    x_thresh_ratio=0.18,
    iou_thresh=0.60
)


# =========================================================
# 5. SPATIAL ASSOCIATION: PERSONS TO MOTORCYCLES
# =========================================================

for person in persons:
    px1, py1, px2, py2 = person["box"]
    person_center_x = (px1 + px2) / 2.0
    person_bottom_y = py2

    matched_bike = None
    max_overlap = 0

    for bike in motorcycles:
        bx1, by1, bx2, by2 = bike["box"]
        bike_width = bx2 - bx1

        # Horizontal alignment check
        is_x_aligned = (
            bx1 - 0.10 * bike_width
            <= person_center_x
            <= bx2 + 0.10 * bike_width
        )

        # Vertical alignment check: Lower body touches/overlaps bike bounding box
        is_y_aligned = (
            person_bottom_y >= by1
            and py1 <= by2
        )

        if is_x_aligned and is_y_aligned:
            overlap_x = max(0, min(px2, bx2) - max(px1, bx1))
            if overlap_x > max_overlap:
                max_overlap = overlap_x
                matched_bike = bike

    if matched_bike is not None:
        matched_bike["riders"].append(person)


# =========================================================
# 6. HELMET DETECTION
# =========================================================

print("[2/4] Running helmet detection...")

helmet_results = helmet_model(
    image_path,
    conf=0.30,
    verbose=False
)

raw_with_helmet = 0
raw_without_helmet = 0
helmet_boxes_data = []

for result in helmet_results:
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = helmet_model.names[cls].lower().strip()
        coords = list(map(int, box.xyxy[0]))

        is_violation = ("without" in label or "no" in label)

        if is_violation:
            raw_without_helmet += 1
        else:
            raw_with_helmet += 1

        helmet_boxes_data.append({
            "box": coords,
            "conf": conf,
            "is_violation": is_violation,
            "label": label
        })


# =========================================================
# 7. STRICT METRIC CALCULATION & CLAMPING
# =========================================================

motorcycle_count = len(motorcycles)
total_associated_riders = sum(len(m["riders"]) for m in motorcycles)

# If riders were matched to bike, use verified rider count; else fallback to raw persons
if total_associated_riders > 0:
    actual_riders = total_associated_riders
    # Helmet violations can NEVER exceed actual riders physically on the motorcycle
    verified_without_helmet = min(raw_without_helmet, actual_riders)
    verified_with_helmet = max(0, actual_riders - verified_without_helmet)
else:
    actual_riders = len(persons)
    verified_without_helmet = raw_without_helmet
    verified_with_helmet = raw_with_helmet

# Determine Violation States
triple_riding = (motorcycle_count > 0 and actual_riders >= 3)
helmet_violation = (verified_without_helmet > 0)
violation = (triple_riding or helmet_violation)

final_status = "VIOLATION DETECTED" if violation else "NO VIOLATION"


# =========================================================
# 8. VISUAL ANNOTATIONS
# =========================================================

print("[3/4] Generating visual annotations...")

# 1. Draw Motorcycle Boxes (Blue)
for i, motorcycle in enumerate(motorcycles, start=1):
    x1, y1, x2, y2 = motorcycle["box"]
    rider_count = len(motorcycle["riders"])

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    draw_label(
        img,
        f"Motorcycle #{i} | Riders: {rider_count}",
        x1 + 5,
        y1 + 22,
        (255, 0, 0)
    )

# 2. Draw Associated Rider Boxes (Green)
for m in motorcycles:
    for p in m["riders"]:
        x1, y1, x2, y2 = p["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        draw_label(
            img,
            f"Rider {p['conf']:.2f}",
            x1 + 5,
            y1 + 20,
            (0, 160, 0),
            scale=0.45
        )

# 3. Draw Helmet Bounding Boxes
for h in helmet_boxes_data:
    x1, y1, x2, y2 = h["box"]
    conf = h["conf"]
    if h["is_violation"]:
        color = (0, 0, 220)
        text = f"NO HELMET {conf:.2f}"
    else:
        color = (0, 180, 0)
        text = f"HELMET {conf:.2f}"

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    draw_label(img, text, x1, max(y1 - 6, 20), color, scale=0.48)


# 4. Report Overlay Card
cv2.rectangle(img, (15, 15), (460, 220), (255, 255, 255), -1)
cv2.rectangle(img, (15, 15), (460, 220), (60, 60, 60), 2)

cv2.putText(img, "TWO-WHEELER VIOLATION REPORT", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 2)
cv2.putText(img, f"Motorcycles: {motorcycle_count}", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1)
cv2.putText(img, f"Riders on Bike: {actual_riders}", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1)
cv2.putText(
    img,
    f"Triple Riding: {'YES' if triple_riding else 'NO'}",
    (25, 120),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.50,
    (0, 0, 255) if triple_riding else (0, 140, 0),
    2
)
cv2.putText(img, f"With Helmet: {verified_with_helmet}", (25, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 140, 0), 2)
cv2.putText(
    img,
    f"Without Helmet: {verified_without_helmet}",
    (25, 170),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.50,
    (0, 0, 255) if helmet_violation else (0, 140, 0),
    2
)
cv2.putText(
    img,
    final_status,
    (25, 202),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (0, 0, 255) if violation else (0, 140, 0),
    2
)


# =========================================================
# 9. SAVE ANNOTATED IMAGE
# =========================================================

output_path = "final_violation_output.jpg"
cv2.imwrite(output_path, img)


# =========================================================
# 10. TERMINAL DETECTION REPORT
# =========================================================

print("\n" + "=" * 60)
print("             DETECTION SUMMARY REPORT")
print("=" * 60)
print(f"Persons Detected       : {len(persons)}")
print(f"Motorcycles Detected   : {motorcycle_count}")
print(f"Associated Riders      : {actual_riders}")
print(f"With Helmet            : {verified_with_helmet}")
print(f"Without Helmet         : {verified_without_helmet}")
print(f"Triple Riding          : {'YES' if triple_riding else 'NO'}")
print(f"Helmet Violation       : {'YES' if helmet_violation else 'NO'}")
print(f"FINAL STATUS           : {final_status}")
print("-" * 60)
print(f"Annotated Image        : {output_path}")


# =========================================================
# 11. LICENSE PLATE OCR + E-CHALLAN GENERATION
# =========================================================

if violation:
    print("\n[4/4] Triggering automated e-Challan generation pipeline...")

    # Use plate crop image if present, else full image
    plate_image = "images.jpg.jpg" if os.path.exists("images.jpg.jpg") else img

    try:
        detected_plate, ocr_conf = extract_plate_number(plate_image)
    except Exception as e:
        print(f"OCR Error: {e}")
        detected_plate = "UNKNOWN"
        ocr_conf = 0.0

    if detected_plate != "UNKNOWN":
        vehicle_registration = detected_plate
        print(f"License Plate Detected : {vehicle_registration}")
        print(f"OCR Confidence         : {ocr_conf:.2f}")
    else:
        vehicle_registration = "PLATE_NOT_DETECTED"
        print("License Plate Detected : NO")
        print("OCR Confidence         : 0.00")

    print("\nGenerating e-Challan...")

    try:
        challan_summary = calculate_and_generate_challan(
            plate_number=vehicle_registration,
            person_count=actual_riders,
            without_helmet_count=verified_without_helmet,
            output_image_path=output_path,
            save_receipt=True
        )
        print("e-Challan generated successfully.")
    except Exception as e:
        print(f"e-Challan generation error: {e}")

else:
    print("\n[4/4] No violations detected.")
    print("System status: NORMAL")


# =========================================================
# 12. DISPLAY RESULT
# =========================================================

print("\n" + "=" * 60)
print("       TWO-WHEELER VIOLATION SYSTEM COMPLETE")
print("=" * 60)
print(f"Final Status : {final_status}")
print(f"Evidence     : {output_path}")
print("=" * 60)

cv2.imshow("Two-Wheeler Violation Detection System", img)
cv2.waitKey(0)
cv2.destroyAllWindows()