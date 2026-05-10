"""
Safety Detection — Image Detection + Ollama Report
Detects helmets and safety vests, generates audit report via LLM
"""

from ultralytics import YOLO
import supervision as sv
import cv2
import requests
import os
import matplotlib.pyplot as plt


MODEL_PATH  = "models/best.pt"
OLLAMA_URL  = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:1b"
CONF_THRESHOLD = 0.4


# ─────────────────────────────────────────
# 1. Load Model
# ─────────────────────────────────────────
def load_model(model_path: str = MODEL_PATH) -> YOLO:
    model = YOLO(model_path)
    model.to("cpu")
    print(f"Model loaded ✅ — {model_path}")
    return model


# ─────────────────────────────────────────
# 2. Run Detection on Image
# ─────────────────────────────────────────
def detect_image(model: YOLO, image_path: str, conf: float = CONF_THRESHOLD):
    results = model.predict(source=image_path, conf=conf, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])

    labels = [
        f"{model.names[class_id]} {confidence:.0%}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    return detections, labels, results[0]


# ─────────────────────────────────────────
# 3. Annotate Image
# ─────────────────────────────────────────
def annotate(image_path: str, detections, labels: list) -> any:
    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    box_annotator   = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    annotated = box_annotator.annotate(frame.copy(), detections)
    annotated = label_annotator.annotate(annotated, detections, labels)
    return annotated


# ─────────────────────────────────────────
# 4. Generate Safety Report via Ollama
# ─────────────────────────────────────────
def generate_report(labels: list) -> str:
    if not labels:
        return "No objects detected in this image."

    detection_text = ", ".join(labels)

    prompt = f"""You are a professional construction site safety inspector.

The following objects were detected on a construction site image:
{detection_text}

Write a concise safety audit report (3-4 sentences) that includes:
- Number of workers and equipment detected
- Safety violations found (NO-Hardhat, NO-Safety Vest)
- Risk level: LOW / MEDIUM / HIGH
- Recommended immediate action

Be professional and concise."""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=30,
        )
        return response.json()["response"]
    except Exception as e:
        return f"Report generation failed: {e}"


# ─────────────────────────────────────────
# 5. Display Results
# ─────────────────────────────────────────
def display(annotated_image, report: str, labels: list):
    plt.figure(figsize=(12, 7))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title("Safety Detection", fontsize=14)
    plt.tight_layout()
    plt.show()

    print("\n📋 SAFETY AUDIT REPORT")
    print("=" * 50)
    print(report)
    print("=" * 50)
    print(f"\n🔍 Detections: {', '.join(labels) if labels else 'None'}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Safety Detection")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to model")
    parser.add_argument("--conf",  type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    args = parser.parse_args()

    model = load_model(args.model)
    detections, labels, _ = detect_image(model, args.image, args.conf)
    annotated  = annotate(args.image, detections, labels)
    report     = generate_report(labels)
    display(annotated, report, labels)
