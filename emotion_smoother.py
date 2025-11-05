#Emotion smoother reads live webcan feed, records raw detections
#computes exponentially smoothed confidence values
#outputs both emotion_raw.json and emotion_smoothed.json
# it smooths out emotions so that transient changes are less likely to be shown


from deepface import DeepFace
import cv2, json, time
from datetime import datetime
import numpy as np
from collections import defaultdict

def exponential_smooth(prev_scores, new_scores, alpha=0.3):
    """Blend new emotion confidences with previous smoothed scores."""
    smoothed = {}
    for emotion, new_conf in new_scores.items():
        old_conf = prev_scores.get(emotion, 0.0)
        smoothed[emotion] = alpha * new_conf + (1 - alpha) * old_conf
    return smoothed

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

    print("Running emotion smoother... press 'q' to quit.")
    raw_log = []
    smooth_log = []

    smoothed_scores = defaultdict(float)
    alpha = 0.25  # smoothing factor; higher = more responsive
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # Only analyze every 10th frame
        if frame_count % 10 == 0:
            result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False, silent=True)[0]
            ...

        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False, silent=True)[0]
        emotion = result["dominant_emotion"]
        confidence = float(result["emotion"][emotion])
        timestamp = datetime.now().strftime("%H:%M:%S")

        # --- RAW LOGGING ---
        raw_log.append({"time": timestamp, "emotion": emotion, "confidence": confidence})

        # --- SMOOTHED UPDATE ---
        smoothed_scores = exponential_smooth(smoothed_scores, result["emotion"], alpha)
        smoothed_dominant = max(smoothed_scores, key=smoothed_scores.get)
        smoothed_conf = smoothed_scores[smoothed_dominant]

        # --- Display ---
        cv2.putText(frame, f"{smoothed_dominant} ({smoothed_conf:.2f}%)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
        cv2.imshow("Smoothed Emotion Detection", frame)

        print(f"Raw: {emotion:8s} ({confidence:5.1f}%)  |  Smoothed: {smoothed_dominant:8s} ({smoothed_conf:5.1f}%)")

        smooth_log.append({
            "time": timestamp,
            "dominant_emotion": smoothed_dominant,
            "confidence": round(float(smoothed_conf), 2),
            "smoothed_scores": {k: round(float(v), 2) for k, v in smoothed_scores.items()}
        })

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.25)

    cap.release()
    cv2.destroyAllWindows()

    # --- Save Logs ---
    with open("emotion_raw.json", "w") as f:
        json.dump(raw_log, f, indent=4)
    with open("emotion_smoothed.json", "w") as f:
        json.dump(smooth_log, f, indent=4)
    print(f"Saved {len(raw_log)} raw and {len(smooth_log)} smoothed entries.")

if __name__ == "__main__":
    main()
