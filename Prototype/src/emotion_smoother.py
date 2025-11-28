from deepface import DeepFace
import cv2, json, time
from datetime import datetime
from collections import defaultdict
from pythonosc import udp_client

# OSC client for Unreal Engine
client = udp_client.SimpleUDPClient("192.168.1.159", 9000)

def exponential_smooth(prev_scores, new_scores, alpha=0.30):
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
    raw_log, smooth_log = [], []
    smoothed_scores = defaultdict(float)
    alpha = 0.30
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # only analyze every 10th frame
        if frame_count % 10 != 0:
            cv2.imshow("Smoothed Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # deepface with crash protection
        try:
            result = DeepFace.analyze(frame, actions=["emotion"],
                                      enforce_detection=False, silent=True)[0]
        except Exception as e:
            print("[DeepFace ERROR]", e)
            continue

        emotion = result["dominant_emotion"]
        confidence = float(result["emotion"][emotion])
        timestamp = datetime.now().strftime("%H:%M:%S")

        # logs
        raw_log.append({"time": timestamp, "emotion": emotion, "confidence": confidence})

        # smooth
        smoothed_scores = exponential_smooth(smoothed_scores, result["emotion"], alpha)
        smoothed_dominant = max(smoothed_scores, key=smoothed_scores.get)
        smoothed_conf = smoothed_scores[smoothed_dominant]

        # send OSC
        client.send_message("/emotion/label", smoothed_dominant)
        client.send_message("/emotion/confidence", float(smoothed_conf))
        for emo, val in smoothed_scores.items():
            client.send_message(f"/emotion/weight/{emo}", float(val))

        # display
        cv2.putText(frame, f"{smoothed_dominant} ({smoothed_conf:.2f}%)",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
        cv2.imshow("Smoothed Emotion Detection", frame)

        # update smooth log
        smooth_log.append({
            "time": timestamp,
            "dominant_emotion": smoothed_dominant,
            "confidence": round(float(smoothed_conf), 2),
            "smoothed_scores": {k: round(float(v), 2) for k, v in smoothed_scores.items()}
        })

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)

    # cleanup
    cap.release()
    cv2.destroyAllWindows()

    # logs
    with open("emotion_raw.json", "w") as f:
        json.dump(raw_log, f, indent=4)
    with open("emotion_smoothed.json", "w") as f:
        json.dump(smooth_log, f, indent=4)

    print(f"Saved {len(raw_log)} raw and {len(smooth_log)} smoothed entries.")

if __name__ == "__main__":
    main()
