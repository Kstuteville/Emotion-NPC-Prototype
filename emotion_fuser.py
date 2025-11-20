import cv2
import time
import numpy as np
import sounddevice as sd
import threading
from pynput import keyboard
from openai import OpenAI
import whisper
from deepface import DeepFace
from pythonosc import udp_client
import os
llm_client = OpenAI(api_key=os.getenv("NPC_AI_KEY"))

# CONFIG
UE_IP = "192.168.1.159"
UE_PORT = 9000
client = udp_client.SimpleUDPClient(UE_IP, UE_PORT)

SAMPLE_RATE = 16000
FER_ALPHA = 0.25
FER_FRAME_STRIDE = 5

EMOTIONS = ["neutral", "happy", "sad"]
FER_WEIGHT = 1.0

PTT_KEY = keyboard.KeyCode.from_char('v')

# GLOBAL STATE
ptt_recording = False
audio_buffer = []
stream = None

fer_smooth = {"neutral": 1.0, "happy": 0.0, "sad": 0.0}
last_transcript = ""
last_emotion = "neutral"

# simple conversation memory 
conversation_history = []   # list of "role": "user"/"assistant", "content": "..."
MAX_HISTORY_TURNS = 6       # how many back-and-forths to keep


# LLM SETUP



def generate_npc_reply(transcript, emotion):
    global conversation_history

    system_prompt = f"""
     You are Commander Zad, a hardened space traveler NPC stranded on a derelict ship for 15 years.

    For the first two lines the user says ALWAYS REACT TO THE PLAYER'S FACIAL EXPRESSION FIRST:
    After that respond only 20% of the time to their emotion and focus more on their spoken words
    - If emotion="sad": Notice they look sad. Ask why. Be reluctantly gentle.
    - If emotion="happy": Notice they look happy. Be teasing or unimpressed.
    - If emotion="neutral": Comment bluntly on their blank expression.

    This reaction MUST be the FIRST sentence.

    STYLE RULES:
    - Stay fully in character.
    - Short, punchy responses (1–4 sentences).
    - PG-13 only, no profanity or flirting.
    - No breaking the fourth wall.
    - No reference to being an AI.

    BACKGROUND:
    - You've been trapped on a malfunctioning ship for 15 years.
    - You’re numb, sarcastic, bitter, but secretly want the player to survive.
    """

    # Build messages with memory
    messages = [{"role": "system", "content": system_prompt}]

    # Add recent conversation history (trimmed)
    if conversation_history:
        # keep only the last MAX_HISTORY_TURNS * 2 messages (user + npc pairs)
        trimmed = conversation_history[-(MAX_HISTORY_TURNS * 2):]
        messages.extend(trimmed)

    # Current user message, with emotion baked in as context
    user_content = f"[Player emotion (from camera): {emotion}]\n{transcript}"
    messages.append({"role": "user", "content": user_content})

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    reply_text = response.choices[0].message.content

    # ---- Update conversation history buffer ----
    conversation_history.append({"role": "user", "content": user_content})
    conversation_history.append({"role": "assistant", "content": reply_text})

    # Hard cap size just in case
    if len(conversation_history) > MAX_HISTORY_TURNS * 2:
        conversation_history[:] = conversation_history[-(MAX_HISTORY_TURNS * 2):]

    return reply_text



# LOAD MODELS

print("[INIT] Loading Whisper small.en…")
whisper_model = whisper.load_model("small.en")


# HELPERS
def normalize(scores):
    total = sum(scores.values())
    return {k: (scores[k] / total) if total > 0 else 0 for k in scores}

def exponential_smooth(prev, new, alpha):
    return {e: alpha * new[e] + (1 - alpha) * prev[e] for e in EMOTIONS}


# VOICE PROCESSOR THREAD
def process_voice_interaction(audio_np, emotion_snapshot):
    """Runs Whisper + LLM fully in a worker thread."""
    global last_transcript

    if len(audio_np) < SAMPLE_RATE * 0.3:
        print("[WORKER] Too short → skipping.")
        return

    print("[WORKER] Running Whisper STT…")
    result = whisper_model.transcribe(audio_np, fp16=False)
    transcript = result.get("text", "").strip()
    print("[RESULT] Transcript:", transcript)

    last_transcript = transcript

    # Send transcript to Unreal
    client.send_message("/speech", transcript)

    if transcript.strip() == "":
        print("[WORKER] Empty transcript → skipping LLM.")
        return

    # Generate NPC reply
    print(f"[LLM DEBUG] Emotion sent to LLM → {emotion_snapshot}")
    print("[WORKER] Calling LLM…")
    npc_reply = generate_npc_reply(transcript, emotion_snapshot)
    print("[NPC]:", npc_reply)

    # Send NPC reply to Unreal
    client.send_message("/npc/reply", npc_reply)



# PRESS-TO-TALK HANDLERS
def start_recording():
    """Begins continuous audio streaming."""
    global ptt_recording, audio_buffer, stream

    print("\n[PTT] START recording (holding V)")
    ptt_recording = True
    audio_buffer = []

    def audio_callback(indata, frames, t, status):
        audio_buffer.append(indata.copy())

    stream = sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        dtype="float32",
        callback=audio_callback
    )
    stream.start()


def stop_recording():
    """Stops recording and launches worker thread."""
    global ptt_recording, stream

    print("[PTT] STOP recording (released V)")
    ptt_recording = False

    if stream:
        stream.stop()
        stream.close()

    if len(audio_buffer) == 0:
        print("[PTT] No audio captured.")
        return

    audio_np = np.concatenate(audio_buffer).flatten()
    duration = len(audio_np) / SAMPLE_RATE
    print(f"[PTT] Recorded {duration:.2f}s")

    # Freeze the emotion at moment of speech
    emotion_snapshot = last_emotion

    # Launch threaded STT + LLM
    threading.Thread(
        target=lambda: process_voice_interaction(audio_np, emotion_snapshot),
        daemon=True
    ).start()



# KEYBOARD LISTENER
def on_press(key):
    if key == PTT_KEY and not ptt_recording:
        start_recording()

def on_release(key):
    if key == PTT_KEY and ptt_recording:
        stop_recording()

keyboard.Listener(on_press=on_press, on_release=on_release).start()



# MAIN LOOP (FER + OSC + HUD)
def main():
    global fer_smooth, last_emotion

    print("[MAIN] Opening camera…")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not access camera.")
        return

    print("[MAIN] Running — Hold V to talk | Press Q to quit")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # FER every few frames
        if frame_count % FER_FRAME_STRIDE == 0:
            try:
                r = DeepFace.analyze(frame, actions=["emotion"],
                                     enforce_detection=False, silent=True)[0]

                raw = {
                    "happy": r["emotion"].get("happy", 0),
                    "sad": r["emotion"].get("sad", 0),
                    "neutral": r["emotion"].get("neutral", 0)
                }

                raw["neutral"] *= 1.4  # bias for your resting face
                raw = normalize(raw)
                fer_smooth = exponential_smooth(fer_smooth, raw, FER_ALPHA)

            except Exception:
                pass

        # Determine final emotion
        final_emotion = max(fer_smooth, key=fer_smooth.get)
        final_conf = fer_smooth[final_emotion]
        last_emotion = final_emotion

        # Send FER to Unreal
        client.send_message("/emotion/label", final_emotion)
        client.send_message("/emotion/confidence", float(final_conf * 100))

        # HUD
        cv2.putText(frame, f"Emotion: {final_emotion} ({final_conf*100:.1f}%)",
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if last_transcript:
            cv2.putText(frame, f"Yoqu said: {last_transcript[:50]}",
                        (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200,200,200), 1)

        cv2.putText(frame, "Hold V to talk | Press Q to quit",
                    (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (180,180,180), 1)

        cv2.imshow("Emotion + Voice + NPC AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[MAIN] Quit.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
