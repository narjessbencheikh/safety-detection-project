"""
Safety Detection — Training Script
Fine-tuning YOLOv11 on Construction Site Safety Dataset
"""

from ultralytics import YOLO
from roboflow import Roboflow
import os


# ─────────────────────────────────────────
# 1. Download Dataset from Roboflow
# ─────────────────────────────────────────
def download_dataset(api_key: str) -> str:
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-universe-projects").project(
        "construction-site-safety"
    )
    dataset = project.version(30).download("yolov11")
    print(f"Dataset downloaded → {dataset.location}")
    return dataset.location


# ─────────────────────────────────────────
# 2. Train YOLOv11
# ─────────────────────────────────────────
def train(data_yaml: str, epochs: int = 50, device: str = "cpu"):
    # Load pretrained YOLOv11 nano — Transfer Learning
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=8,            # reduce if RAM issues
        device=device,      # "cuda" if GPU available, else "cpu"
        project="runs/safety",
        name="helmet_vest",
    )

    print(f"\nTraining complete ✅")
    print(f"Best model saved → runs/safety/helmet_vest/weights/best.pt")
    return results


# ─────────────────────────────────────────
# 3. Evaluate Model
# ─────────────────────────────────────────
def evaluate(model_path: str, data_yaml: str):
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)

    print("\n📊 EVALUATION RESULTS")
    print("=" * 40)
    print(f"mAP50      : {metrics.box.map50:.3f}")
    print(f"mAP50-95   : {metrics.box.map:.3f}")
    print(f"Precision  : {metrics.box.mp:.3f}")
    print(f"Recall     : {metrics.box.mr:.3f}")
    print("=" * 40)
    return metrics


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    API_KEY = "YOUR_ROBOFLOW_API_KEY"   # ← replace with your key

    dataset_path = download_dataset(API_KEY)
    data_yaml = os.path.join(dataset_path, "data.yaml")

    train(data_yaml=data_yaml, epochs=50, device="cpu")

    evaluate(
        model_path="runs/safety/helmet_vest/weights/best.pt",
        data_yaml=data_yaml,
    )
