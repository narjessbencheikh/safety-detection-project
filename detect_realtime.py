"""
Safety Detection — Real-Time Detection
Webcam or video file with live helmet/vest detection + Ollama reports
"""

from ultralytics import YOLO
import supervision as sv
import cv2
import requests
import time
import argparse


MODEL_PATH   = "models/best.pt"
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:1b"
CONF_THRESHOLD = 0.4
REPORT_INTERVAL = 60    # generate report every 60 frames


# ─────────────────────────────────────────
# Ollama Report (runs fast, one sentence)
# ─────────────────────────────────────────
def quick_report(labels: list) -> str:
    if not labels:
        return "No objects detected."

    detection_text = ", ".join(labels)
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": f"Safety inspector. Detected: {detection_text}. One sentence risk level and action.",
                "stream": False,
            },
            timeout=10,
        )
        return response.json()["response"].strip()
    except Exception:
        return "Ollama not available."


# ─────────────────────────────────────────
# Draw Status Overlay on Frame
# ─────────────────────────────────────────
def draw_overlay(frame, labels: list, report: str, fps: float):
    h, w = frame.shape[:2]

    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Violation alert
    violations = [l for l in labels if "NO-" in l]
    if violations:
        cv2.putText(frame, f"⚠ VIOLATION: {len(violations)} found",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "✓ No violations detected",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Report overlay at bottom
    if report:
        report_short = report[:80] + "..." if len(report) > 80 else report
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, report_short, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return frame


# ─────────────────────────────────────────
# Main Real-Time Loop
# ─────────────────────────────────────────
def run(source=0, model_path=MODEL_PATH, conf=CONF_THRESHOLD):
    model = YOLO(model_path)
    model.to("cpu")

    box_annotator   = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    # 0 = webcam | "path/to/video.mp4" = video file
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Cannot open source: {source}")
        return

    print("Safety Detection LIVE — press 'q' to quit\n")

    frame_count = 0
    report      = ""
    prev_time   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Detection ──────────────────────
        results    = model.predict(frame, conf=conf, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])

        labels = [
            f"{model.names[c]} {conf_:.0%}"
            for c, conf_ in zip(detections.class_id, detections.confidence)
        ]

        # ── Annotate ───────────────────────
        annotated = box_annotator.annotate(frame.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, labels)

        # ── FPS ────────────────────────────
        curr_time = time.time()
        fps       = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # ── Ollama report every N frames ───
        if frame_count % REPORT_INTERVAL == 0 and labels:
            report = quick_report(labels)
            print(f"\n📋 [{frame_count}] {report}")

        # ── Overlay ────────────────────────
        annotated = draw_overlay(annotated, labels, report, fps)

        cv2.imshow("Safety Detection — LIVE (q to quit)", annotated)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended — {frame_count} frames processed ✅")


# ─────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Safety Detection")
    parser.add_argument("--source", default="0",       help="0=webcam or path to video")
    parser.add_argument("--model",  default=MODEL_PATH, help="Path to model weights")
    parser.add_argument("--conf",   type=float, default=CONF_THRESHOLD)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run(source=source, model_path=args.model, conf=args.conf)
