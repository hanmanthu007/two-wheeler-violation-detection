from google import genai
from PIL import Image
import os
import re
import time


# =========================================================
# GEMINI API CONFIGURATION
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )

client = genai.Client(api_key=GEMINI_API_KEY)


# =========================================================
# GEMINI VEHICLE VERIFICATION
# =========================================================

def verify_with_gemini(image_input):
    """
    Sends a vehicle image to Gemini to extract the license
    plate number and provide a visual audit explanation.
    """

    # -----------------------------------------------------
    # Handle image path
    # -----------------------------------------------------

    if isinstance(image_input, str):

        if not os.path.exists(image_input):
            print(f"[!] File not found: {image_input}")
            return "UNKNOWN", "Image file not found."

        pil_img = Image.open(image_input).convert("RGB")

    # -----------------------------------------------------
    # Handle OpenCV image
    # -----------------------------------------------------

    else:

        import cv2

        rgb_img = cv2.cvtColor(
            image_input,
            cv2.COLOR_BGR2RGB
        )

        pil_img = Image.fromarray(rgb_img)

    # =====================================================
    # GEMINI PROMPT
    # =====================================================

    prompt = """
Analyze this two-wheeler traffic image.

Return exactly two lines:

PLATE: <license plate number or UNKNOWN>
REASON: <brief helmet observation>

Rules:
- Use only alphanumeric characters for the plate.
- If the plate cannot be read clearly, use UNKNOWN.
- State whether the visible motorcycle riders are wearing helmets.
- Do not add any extra text.
"""

    # =====================================================
    # GEMINI API REQUEST WITH RETRY
    # =====================================================

    max_retries = 5

    for attempt in range(1, max_retries + 1):

        try:

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=[
                    pil_img,
                    prompt
                ]
            )

            text = (response.text or "").strip()

            print(f"[+] Gemini response received on attempt {attempt}")

            # =================================================
            # PARSE GEMINI RESPONSE
            # =================================================

            plate_match = re.search(
                r"PLATE:\s*([A-Za-z0-9]+)",
                text,
                re.IGNORECASE
            )

            reason_match = re.search(
                r"REASON:\s*(.+)",
                text,
                re.IGNORECASE
            )

            if plate_match:
                detected_plate = plate_match.group(1).upper()
            else:
                detected_plate = "UNKNOWN"

            if reason_match:
                explanation = reason_match.group(1).strip()
            else:
                explanation = "Visual audit completed."

            return detected_plate, explanation

        except Exception as e:

            error_message = str(e)

            # -------------------------------------------------
            # Temporary Gemini service error
            # -------------------------------------------------

            if "503" in error_message or "UNAVAILABLE" in error_message:

                if attempt < max_retries:

                    wait_time = attempt * 5

                    print(
                        f"[!] Gemini service temporarily unavailable. "
                        f"Retrying ({attempt}/{max_retries}) "
                        f"in {wait_time}s..."
                    )

                    time.sleep(wait_time)

                else:

                    print(
                        "[!] Gemini service is currently unavailable "
                        "after multiple attempts."
                    )

                    return (
                        "UNKNOWN",
                        "Gemini service temporarily unavailable."
                    )

            # -------------------------------------------------
            # Other API errors
            # -------------------------------------------------

            else:

                print(
                    f"[!] Gemini API Verification Error: {e}"
                )

                return (
                    "UNKNOWN",
                    "API verification failed."
                )


# =========================================================
# STANDALONE TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 50)
    print("      GEMINI MULTIMODAL VERIFIER TEST")
    print("=" * 50)

    # -----------------------------------------------------
    # Select available test image
    # -----------------------------------------------------

    if os.path.exists("images.jpg.jpg"):

        test_img = "images.jpg.jpg"

    elif os.path.exists("test.jpg.jpg"):

        test_img = "test.jpg.jpg"

    else:

        print("[!] No test image found.")
        print("Please place a test image in the project folder.")
        exit()

    print(f"Testing on: {test_img}")
    print()

    # -----------------------------------------------------
    # Run Gemini verification
    # -----------------------------------------------------

    plate, reason = verify_with_gemini(test_img)

    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    print("=" * 50)
    print(f"Detected Plate : {plate}")
    print(f"Audit Finding  : {reason}")
    print("=" * 50)