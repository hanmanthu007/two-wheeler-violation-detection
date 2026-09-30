# Two-Wheeler Traffic Violation Detection System

An AI-powered computer vision project for detecting common two-wheeler traffic violations from vehicle images using YOLOv8, helmet detection, rider-motorcycle association, license plate OCR, and automated e-Challan report generation.

## Features

- Motorcycle and person detection using YOLOv8
- Rider-to-motorcycle association using spatial analysis
- Duplicate rider filtering
- Helmet and no-helmet detection
- Triple-riding detection
- Traffic violation identification
- License plate extraction using EasyOCR
- Automated e-Challan-style text report generation
- Annotated violation evidence images
- Optional Gemini-based multimodal vehicle verification

## System Workflow

```text
Input Vehicle Image
        ↓
YOLOv8 Person & Motorcycle Detection
        ↓
Rider-Motorcycle Association
        ↓
Helmet Detection + Rider Counting
        ↓
Violation Analysis
        ↓
License Plate OCR
        ↓
e-Challan Report Generation
        ↓
Annotated Evidence Output

## Demo / Results

The system was tested on two-wheeler traffic images to identify riders, motorcycles, helmet compliance, and common traffic violations.

### Example Detection Result

For a test image containing three riders on a single motorcycle, the system produced the following detection summary:

| Detection | Result |
|---|---|
| Persons Detected | 3 |
| Motorcycles Detected | 1 |
| Associated Riders | 3 |
| Riders With Helmet | 0 |
| Riders Without Helmet | 3 |
| Triple Riding | Yes |
| Helmet Violation | Yes |
| Final Status | Violation Detected |
| License Plate OCR | Not Detected |

The system also generates an annotated evidence image and a simulated e-Challan-style text report when a violation is detected.

> **Note:** The e-Challan output is a simulated demonstration report and is not connected to any government traffic-management or e-Challan service.

## Sample Output

The annotated output highlights:

- Rider detection
- Motorcycle detection
- Rider-to-motorcycle association
- Helmet / no-helmet classification
- Triple-riding detection
- Violation status