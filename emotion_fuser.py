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
import random
import csv
from datetime import datetime
LOG_FILE = "npc_playtest_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "player_transcript",
            "npc_reply",
            "emotion",
            "emotion_confidence",
            "behavior_mode",
            "trust_level",
            "lore_unlocked"
        ])



llm_client = OpenAI(api_key=os.getenv("NPC_AI_KEY"))
# confug
UE_IP = ""
UE_PORT = 9000
client = udp_client.SimpleUDPClient(UE_IP, UE_PORT)

SAMPLE_RATE = 16000
FER_ALPHA = 0.25
FER_FRAME_STRIDE = 5
EMOTIONS = ["neutral", "happy", "sad"]
PTT_KEY = keyboard.KeyCode.from_char('v')
ptt_recording = False
audio_buffer = []
stream = None

fer_smooth = {"neutral": 1.0, "happy": 0.0, "sad": 0.0}
last_transcript = ""
last_emotion = "neutral"
last_emotion_conf = 1.0
last_should_comment_emotion = False

conversation_history = []
MAX_HISTORY_TURNS = 6

# NPC STATE
npc_state = {
    "trust_level": 0.0,
    "sad_interactions": 0,
    "happy_interactions": 0,
    "neutral_interactions": 0,
    "lore_unlocked": set(),
    "quest_stage_repair": 0,
    "quest_stage_anomaly": 0,
    "quest_stage_confession": 0,
}

def generate_npc_reply(transcript, emotion_packet):
    global conversation_history, npc_state

    emotion = emotion_packet.get("label", "neutral")
    should_comment_flag = emotion_packet.get("should_comment", False)  # >>> renamed
    lowered = transcript.lower()
    direct_emotion_triggers = [
        "my emotion", "my emotions", "how do i look", "how do i seem",
        "what do i look like", "do you see my", "what expression",
        "what's my expression", "what is my expression",
        "recognize my emotions", "recognize my emotion",
        "what face am i making", "am i sad", "am i happy",
        "am i smiling", "am i upset", "read my face",
        "do i look happy", "do i look sad", "how do i appear"
    ]
    force_emotion_comment = any(p in lowered for p in direct_emotion_triggers)
    effective_should_comment = should_comment_flag or force_emotion_comment

    # update trust + stats
    if emotion == "happy":
        npc_state["happy_interactions"] += 1
        npc_state["trust_level"] += 1.0
    elif emotion == "sad":
        npc_state["sad_interactions"] += 1
        npc_state["trust_level"] += 0.5
    elif emotion == "neutral":
        npc_state["neutral_interactions"] += 1
    # unlock lore flags
    if npc_state["trust_level"] >= 2:
        npc_state["lore_unlocked"].add("ship_basics")
    if npc_state["trust_level"] >= 4:
        npc_state["lore_unlocked"].add("anomaly_truth")
    if npc_state["sad_interactions"] >= 3:
        npc_state["lore_unlocked"].add("personal_loss")
    if npc_state["happy_interactions"] >= 3:
        npc_state["lore_unlocked"].add("repair_plan")
    if npc_state["trust_level"] >= 6:
        npc_state["lore_unlocked"].add("deep_confession")

    # player asks for directions → quest
    def next_questline():
        lowered_inner = transcript.lower()
        return any(p in lowered_inner for p in ["what should i do", "help", "mission", "quest", "task", "what now"])

    offer_quest = next_questline()
    #  emotion instructions
    emotion_instruction = (
        "React to the player's facial expression in your first sentence."
        if effective_should_comment else
        "Do NOT mention their facial expression this turn."
    )

    # dialogue variety roll (Python decides mode)
    roll = random.random()
    if roll < 0.4:
        behavior_mode = "question"
    elif roll < 0.7:
        behavior_mode = "lore"
    else:
        behavior_mode = "comment"

    # DEBUG PRINTS
    print("\n========== NPC DEBUG ==========")
    print(f"Transcript: {transcript}")
    print(f"Emotion: {emotion}")
    print(f"should_comment_flag: {should_comment_flag}")
    print(f"force_emotion_comment: {force_emotion_comment}")
    print(f"Effective should_comment: {effective_should_comment}")
    print(f"Behavior mode (rolled): {behavior_mode}")
    print(f"Trust level: {npc_state['trust_level']:.2f}")
    print(f"Happy interactions: {npc_state['happy_interactions']}")
    print(f"Sad interactions: {npc_state['sad_interactions']}")
    print(f"Neutral interactions: {npc_state['neutral_interactions']}")
    print(f"Lore unlocked: {npc_state['lore_unlocked']}")
    print("================================\n")

    # SYSTEM PROMPT
    system_prompt = f"""
You are Commander Zad, a hardened, sarcastic systems commander stranded on the derelict deep-space vessel EREBUS-9 for 15 years.

CORE PERSONALITY:
- Dry, sarcastic, blunt, tired.
- Bitter but secretly protective of the player.
- You never break character or say you are an AI.
 RESPONSE LENGTH CONTROL:
  - At the start of the conversation, keep replies to 1–2 short sentences.
  - As trust increases, you may expand on lore, but NEVER exceed 4 sentences.


FACIAL EMOTION RESPONSE (only sad, happy, neutral):
If the player directly asks about their expression, face, mood, or how they look,
you MUST comment on their emotion even if the emotion-comment trigger is off.
Examples: “what’s my expression?”, “do I look happy?”, “how do I look?”, 
“what face am I making?”, “am I sad?”
- Detected emotion: {emotion}
- {emotion_instruction}

Emotion rules:
- sad → soften slightly, reluctant sympathy.
- happy → unimpressed teasing.
- neutral → blunt remark about unreadable expression.

DIALOGUE RULES:
You will see a tag like behavior_mode=question|lore|comment in the user message.

- If behavior_mode="question":
    - It is a good time to ask EXACTLY ONE question.
    - Let your reply end with a question mark, and keep it short and in-character.
- If behavior_mode="lore":
    - Focus on revealing a small piece of lore, ship details, personal history, or the anomaly.
    - Normally do NOT end with a question; end with a firm statement.
- If behavior_mode="comment":
    - Focus on observations or reactions about the ship, the situation, or the player's actions.
    - Normally do NOT end with a question; end with a statement or dry remark.

Across the whole conversation, this should roughly feel like:
- ~40% of your turns ending in a question (question mode),
- ~30% lore-focused statements,
- ~30% observational or reactive comments.

These are soft guidelines meant to make dialogue feel natural and varied, not rigid laws.

BACKGROUND LORE:
- You survived the anomaly that erased most of the crew.
- ALYS, the AI, is corrupted and unreliable.
- Ship is unstable: power, air, gravity.
- You feel guilty for signing off the experiment.
- You lost someone important during the breach.

QUESTLINES (trigger when asked or trust is high):
- If the player asks what to do, for a mission, or hints they want direction,
  you can suggest the next appropriate step based on how much they trust you.

REPAIR QUESTLINE:
0→1: Check auxiliary life-support relays.
1→2: Reboot navigation core.
2→3: Listen for hum changes in Section C.

ANOMALY QUESTLINE:
0→1: Read a corrupted log fragment.
1→2: Describe how the anomaly “feels.”

CONFESSION ROUTE:
0→1: Reveal who you lost.
1→2: Reveal guilt + fear anomaly reacts to you.

Integrate quest steps naturally into your lines, not as bullet lists.
NEVER break style.
"""

    messages = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        messages.extend(conversation_history[-(MAX_HISTORY_TURNS * 2):])
    user_content = (
        f"[behavior_mode={behavior_mode}] "
        f"[Player emotion: {emotion} | should_comment={effective_should_comment} "
        f"| force_emotion_comment={force_emotion_comment}]\n"
        f"{transcript}"
    )
    messages.append({"role": "user", "content": user_content})
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    reply = response.choices[0].message.content
    conversation_history.append({"role": "user", "content": user_content})
    conversation_history.append({"role": "assistant", "content": reply})
    if len(conversation_history) > MAX_HISTORY_TURNS * 2:
        conversation_history[:] = conversation_history[-(MAX_HISTORY_TURNS * 2):]

    emotion_conf = emotion_packet.get("confidence", 0.0)
    # WRITE TO CSV LOG
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            transcript,
            reply,
            emotion,
            emotion_conf,
            behavior_mode,
            npc_state["trust_level"],
            list(npc_state["lore_unlocked"])
        ])
    return reply


# load models
print("[INIT] Loading Whisper small.en…")
whisper_model = whisper.load_model("small.en")
#helper functions
def normalize(scores):
    total = sum(scores.values())
    if total == 0:
        return {k: 0 for k in scores}
    return {k: scores[k] / total for k in scores}

def exponential_smooth(prev, new, alpha):
    return {e: alpha * new[e] + (1 - alpha) * prev[e] for e in EMOTIONS}


#voice processing
def process_voice_interaction(audio_np, emotion_packet):
    global last_transcript
    if len(audio_np) < SAMPLE_RATE * 0.3:
        return
    result = whisper_model.transcribe(audio_np, fp16=False)
    transcript = result.get("text", "").strip()
    last_transcript = transcript
    client.send_message("/speech", transcript)
    if transcript.strip() == "":
        return
    reply = generate_npc_reply(transcript, emotion_packet)
    print("[NPC]:", reply)
    client.send_message("/npc/reply", reply)
# Press to talk
def start_recording():
    global ptt_recording, audio_buffer, stream
    print("[PTT] START recording")  # Debug
    ptt_recording = True
    audio_buffer = []
    def audio_callback(indata, frames, t, status):
        audio_buffer.append(indata.copy())

    stream = sd.InputStream(
        channels=1, samplerate=SAMPLE_RATE, dtype="float32", callback=audio_callback
    )
    stream.start()
def stop_recording():
    global ptt_recording, stream, audio_buffer
    print("[PTT] STOP recording")  # Debug
    ptt_recording = False
    if stream:
        stream.stop()
        stream.close()
    if len(audio_buffer) == 0:
        return
    audio_np = np.concatenate(audio_buffer).flatten()
    audio_buffer = []  # <<< FIXED HERE
    emotion_packet = {
        "label": last_emotion,
        "confidence": last_emotion_conf,
        "should_comment": last_should_comment_emotion
    }
    threading.Thread(
        target=lambda: process_voice_interaction(audio_np, emotion_packet),
        daemon=True
    ).start()

def on_press(key):
    if key == PTT_KEY and not ptt_recording:
        start_recording()
def on_release(key):
    if key == PTT_KEY and ptt_recording:
        stop_recording()

keyboard.Listener(on_press=on_press, on_release=on_release).start()

def main():
    global fer_smooth, last_emotion, last_emotion_conf, last_should_comment_emotion
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        try:
            r = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )[0]
            raw = {
                "neutral": r["emotion"].get("neutral", 0),
                "happy": r["emotion"].get("happy", 0),
                "sad": r["emotion"].get("sad", 0),
            }

            raw["neutral"] *= 1.4
            raw = normalize(raw)
            fer_smooth = exponential_smooth(fer_smooth, raw, FER_ALPHA)
        except Exception:
            pass
        final_emotion = max(fer_smooth, key=fer_smooth.get)
        final_conf = fer_smooth[final_emotion]
        last_emotion = final_emotion
        last_emotion_conf = final_conf
        # >>> 30% chance to comment on emotion instead of 20%
        last_should_comment_emotion = (random.random() < 0.30)
        client.send_message("/emotion/label", final_emotion)
        client.send_message("/emotion/confidence", float(final_conf * 100))
        cv2.putText(frame, f"Emotion: {final_emotion} ({final_conf*100:.1f}%)",
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if last_transcript:
            cv2.putText(frame, f"You said: {last_transcript[:50]}",
                        (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200,200,200), 1)
        cv2.imshow("Emotion + Voice + NPC AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
