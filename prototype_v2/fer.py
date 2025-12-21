from deepface import DeepFace
from config import EMOTIONS, FER_ALPHA

fer_smooth = {"neutral": 1.0, "happy": 0.0, "sad": 0.0}

def normalize(scores):
    total = sum(scores.values())
    return {k: scores[k] / total if total else 0 for k in scores}

def exponential_smooth(prev, new, alpha):
    return {e: alpha * new[e] + (1 - alpha) * prev[e] for e in EMOTIONS}

def detect_emotion(frame):
    global fer_smooth
    r = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False, silent=True)[0]

    raw = {
        "neutral": r["emotion"].get("neutral", 0) * 1.4,
        "happy": r["emotion"].get("happy", 0),
        "sad": r["emotion"].get("sad", 0),
    }

    raw = normalize(raw)
    fer_smooth = exponential_smooth(fer_smooth, raw, FER_ALPHA)

    emotion = max(fer_smooth, key=fer_smooth.get)
    return emotion, fer_smooth[emotion]
