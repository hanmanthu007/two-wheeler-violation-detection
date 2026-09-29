from datetime import datetime
import os

# =========================================================
# FINE SLABS & MOTOR VEHICLE ACT RULES
# =========================================================
FINE_STRUCTURE = {
    "TRIPLE_RIDING": 1000,       # Section 128/177 MV Act
    "OVERLOADING": 2000,         # Section 194A MV Act (>3 riders)
    "NO_HELMET_PER_RIDER": 1000  # Section 129/177 MV Act (Per rider)
}


def calculate_and_generate_challan(
    plate_number="UNKNOWN",
    person_count=3,
    without_helmet_count=3,
    output_image_path="final_violation_output.jpg",
    save_receipt=True
):
    """
    Computes total traffic violation fines and generates an e-Challan report.
    """
    challan_id = f"ECH-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    timestamp = datetime.now().strftime("%d-%b-%Y %I:%M:%S %p")
    
    violations_detected = []
    total_fine = 0

    # 1. Overloading / Triple Riding Assessment
    if person_count == 3:
        fine_amount = FINE_STRUCTURE["TRIPLE_RIDING"]
        violations_detected.append(("Triple Riding (3 Persons on 2-Wheeler)", fine_amount))
        total_fine += fine_amount
    elif person_count > 3:
        fine_amount = FINE_STRUCTURE["OVERLOADING"]
        violations_detected.append((f"Dangerous Overloading ({person_count} Persons)", fine_amount))
        total_fine += fine_amount

    # 2. Helmet Violation Assessment
    if without_helmet_count > 0:
        fine_amount = without_helmet_count * FINE_STRUCTURE["NO_HELMET_PER_RIDER"]
        violations_detected.append(
            (f"Riding Without Helmet ({without_helmet_count} Rider{'s' if without_helmet_count > 1 else ''})", fine_amount)
        )
        total_fine += fine_amount

    # 3. Format e-Challan Receipt
    divider = "=" * 55
    sub_divider = "-" * 55

    receipt_lines = [
        divider,
        "          STATE TRAFFIC POLICE E-CHALLAN",
        divider,
        f"Challan ID         : {challan_id}",
        f"Date & Time        : {timestamp}",
        f"Vehicle Number     : {plate_number}",
        f"Riders Detected    : {person_count}",
        sub_divider,
        "VIOLATIONS CHARGED :"
    ]

    if violations_detected:
        for idx, (violation, cost) in enumerate(violations_detected, start=1):
            receipt_lines.append(f" {idx}. {violation:<38} : Rs {cost:>5}")
    else:
        receipt_lines.append(" None (No Violations Recorded)")

    receipt_lines.extend([
        sub_divider,
        f"TOTAL FINE PAYABLE  : Rs {total_fine}",
        f"EVIDENCE SAVED      : {output_image_path}",
        divider
    ])

    challan_text = "\n".join(receipt_lines)
    print("\n" + challan_text + "\n")

    # 4. Save e-Challan to Text Log
    if save_receipt and violations_detected:
        os.makedirs("challans", exist_ok=True)
        log_filename = f"challans/{challan_id}_{plate_number}.txt"
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write(challan_text)
        print(f"Receipt saved to -> {log_filename}")

    return {
        "challan_id": challan_id,
        "plate_number": plate_number,
        "total_fine": total_fine,
        "violations": violations_detected
    }


# =========================================================
# STANDALONE TEST EXECUTION
# =========================================================
if __name__ == "__main__":
    # Test sample execution matching your detection results
    calculate_and_generate_challan(
        plate_number="TS09EA1234",
        person_count=3,
        without_helmet_count=3,
        output_image_path="final_violation_output.jpg"
    )