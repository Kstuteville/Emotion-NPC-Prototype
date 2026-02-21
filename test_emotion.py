from deepface import DeepFace
import cv2, json, time
from datetime import datetime

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return
    log = []
    print("Running... press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # analyze emotion
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False, silent=True)[0]
        emotion = str(result["dominant_emotion"])
        confidence = float(result["emotion"][emotion])
        timestamp = datetime.now().strftime("%H:%M:%S")
        #show on video feed
        cv2.putText(frame, f"{emotion} ({confidence:.2f}%)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
        cv2.imshow("Emotion Detection", frame)
        print(f"[{timestamp}] {emotion} ({confidence:.2f}%)")
        log.append({
            "time": timestamp,
            "emotion": emotion,
            "confidence": round(confidence, 2)
        })
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.2)

    cap.release()
    cv2.destroyAllWindows()
    #ensure all data is JSON-serializable
    safe_log = [{str(k): (float(v) if isinstance(v, (int, float)) else v) for k, v in entry.items()} for entry in log]
    with open("emotion_log.json", "w", encoding="utf-8") as f:
        json.dump(safe_log, f, indent=4)
    print(f"Saved {len(log)} entries to emotion_log.json")
if __name__ == "__main__":
    main()
