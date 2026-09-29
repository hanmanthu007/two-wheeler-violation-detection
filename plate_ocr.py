import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import easyocr
import cv2
import re
import os

print("Initializing EasyOCR reader...")
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR initialized successfully!\n")

# Allowed standard license plate characters
PLATE_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def clean_text(text):
    """Filters out any non-alphanumeric character."""
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def extract_plate_number(image_input, min_conf=0.10):
    """
    Extracts and spatially sequences license plate text from the image.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            print(f"Error: File not found -> {image_input}")
            return "UNKNOWN", 0.0
        img = cv2.imread(image_input)
    else:
        img = image_input

    if img is None:
        print("Error: Could not decode image.")
        return "UNKNOWN", 0.0

    # 1. Run EasyOCR with alphanumeric constraint
    results = reader.readtext(
        img,
        allowlist=PLATE_CHARS,
        paragraph=False,
        text_threshold=0.15,
        low_text=0.20,
        mag_ratio=1.2
    )

    if not results:
        # Fallback to enhanced grayscale if initial pass yields nothing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results = reader.readtext(
            enhanced,
            allowlist=PLATE_CHARS,
            paragraph=False,
            text_threshold=0.15,
            low_text=0.20,
            mag_ratio=1.2
        )

    # 2. Filter and collect character bounding boxes
    valid_detections = []
    print("--- RAW OCR DETECTIONS ---")
    for bbox, raw_text, conf in results:
        cleaned = clean_text(raw_text)
        print(f"Raw: '{raw_text}' -> Cleaned: '{cleaned}' | Conf: {conf:.2f}")

        if len(cleaned) > 0 and conf >= min_conf:
            # bbox is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            top_left_x = bbox[0][0]
            top_left_y = bbox[0][1]
            valid_detections.append({
                "text": cleaned,
                "conf": conf,
                "x": top_left_x,
                "y": top_left_y
            })
    print("--------------------------")

    if not valid_detections:
        return "UNKNOWN", 0.0

    # 3. Spatial Sorting: Top-to-Bottom, then Left-to-Right
    # Group boxes by row proximity (within 25 pixels vertically)
    valid_detections.sort(key=lambda d: (round(d["y"] / 25) * 25, d["x"]))

    # 4. Assemble candidate plate string
    plate_text = "".join([d["text"] for d in valid_detections])
    avg_conf = sum([d["conf"] for d in valid_detections]) / len(valid_detections)

    # Plates should generally have at least 4 characters
    if len(plate_text) >= 4:
        return plate_text, avg_conf

    return "UNKNOWN", 0.0


# =========================================================
# STANDALONE EXECUTION
# =========================================================
if __name__ == "__main__":
    test_image = "images.jpg.jpg"

    plate_no, conf = extract_plate_number(test_image)

    print("\n" + "=" * 45)
    print("        LICENSE PLATE OCR RESULTS")
    print("=" * 45)
    print(f"Plate Number : {plate_no}")
    print(f"Confidence   : {conf:.2f}")
    print("=" * 45)