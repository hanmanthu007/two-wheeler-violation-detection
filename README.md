\# Two-Wheeler Traffic Violation Detection System



An AI-powered computer vision project for detecting common two-wheeler traffic violations from vehicle images using YOLOv8, helmet detection, rider-motorcycle association, license plate OCR, and automated e-Challan report generation.



\## Features



\- Motorcycle and person detection using YOLOv8

\- Rider-to-motorcycle association using spatial analysis

\- Duplicate rider filtering

\- Helmet and no-helmet detection

\- Triple-riding detection

\- Traffic violation identification

\- License plate extraction using EasyOCR

\- Automated e-Challan-style text report generation

\- Annotated violation evidence images

\- Optional Gemini-based multimodal vehicle verification



\## System Workflow



```text

Input Vehicle Image

&#x20;       ↓

YOLOv8 Person \& Motorcycle Detection

&#x20;       ↓

Rider-Motorcycle Association

&#x20;       ↓

Helmet Detection + Rider Counting

&#x20;       ↓

Violation Analysis

&#x20;       ↓

License Plate OCR

&#x20;       ↓

e-Challan Report Generation

&#x20;       ↓

Annotated Evidence Output

