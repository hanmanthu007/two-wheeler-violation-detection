from huggingface_hub import hf_hub_download
import shutil
import os

print("Downloading Helmet Detection Model...")

model_path = hf_hub_download(
    repo_id="iam-tsr/yolov8n-helmet-detection",
    filename="best.pt"
)

os.makedirs("models", exist_ok=True)

shutil.copy(model_path, "models/helmet_model.pt")

print("Helmet model downloaded successfully!")
print("Saved as: models/helmet_model.pt")