import random
import csv
from datetime import datetime
from config import LOG_FILE, MAX_HISTORY_TURNS
from state import npc_state, conversation_history
from llm import llm_client

# --------------------------------------------------
# NPC BRAIN — emotion → trust → dialogue → memory
# --------------------------------------------------

def generate_npc_reply(transcript, emotion_packet):
    """
    Core NPC reasoning function.
    Takes player transcript + emotion packet and returns NPC reply + behavior mode.
    """

    # ---------------- SAFETY GUARD ----------------
    if not transcript or not transcript.strip():
        return None, None

    emotion = emotion_packet.get("label", "neutral")
    confidence = emotion_packet.get("confidence", 0.0)
    should_comment_flag = emotion_packet.get("should_comment", False)
    lowered = transcript.lower()

    # ---------------- EMOTION COMMENT OVERRIDE ----------------
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

    # ---------------- TRUST + STATS ----------------
    if emotion == "happy":
        npc_state["happy_interactions"] += 1
        npc_state["trust_level"] += 1.0
    elif emotion == "sad":
        npc_state["sad_interactions"] += 1
        npc_state["trust_level"] += 0.5
    else:
        npc_state["neutral_interactions"] += 1

    # ---------------- LORE UNLOCKS ----------------
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

    # ---------------- BEHAVIOR MODE ----------------
    roll = random.random()
    if roll < 0.4:
        behavior_mode = "question"
    elif roll < 0.7:
        behavior_mode = "lore"
    else:
        behavior_mode = "comment"

    # ---------------- EMOTION INSTRUCTION ----------------
    emotion_instruction = (
        "React to the player's facial expression in your first sentence."
        if effective_should_comment
        else "Do NOT mention their facial expression this turn."
    )

    # ---------------- SYSTEM PROMPT ----------------
    system_prompt = f"""
You are Commander Zad, a hardened, sarcastic systems commander stranded on the
derelict deep-space vessel EREBUS-9 for 15 years.

CORE PERSONALITY:
- Dry, sarcastic, blunt, tired.
- Bitter but secretly protective of the player.
- You never break character or say you are an AI.

RESPONSE LENGTH CONTROL:
- Early trust: 1–2 short sentences.
- Higher trust: may expand, NEVER exceed 4 sentences.

FACIAL EMOTION RESPONSE:
- Detected emotion: {emotion}
- {emotion_instruction}

Emotion rules:
- sad → soften slightly, reluctant sympathy.
- happy → unimpressed teasing.
- neutral → blunt remark about unreadable expression.

DIALOGUE RULES:
You will see a tag like behavior_mode=question|lore|comment in the user message.

- question → ask EXACTLY one question.
- lore → reveal small lore detail.
- comment → observational or reactive remark.

BACKGROUND:
- You survived the anomaly that erased most of the crew.
- ALYS, the AI, is corrupted.
- You signed off on the experiment.
- You lost someone during the breach.

NEVER break style or acknowledge being an AI.
"""

    # ---------------- MESSAGE BUILD ----------------
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history[-(MAX_HISTORY_TURNS * 2):])

    user_content = (
        f"[behavior_mode={behavior_mode}] "
        f"[Player emotion: {emotion} | should_comment={effective_should_comment} "
        f"| force_emotion_comment={force_emotion_comment}]\n"
        f"{transcript}"
    )

    messages.append({"role": "user", "content": user_content})

    # ---------------- LLM CALL ----------------
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    reply = response.choices[0].message.content

    # ---------------- MEMORY UPDATE ----------------
    conversation_history.append({"role": "user", "content": user_content})
    conversation_history.append({"role": "assistant", "content": reply})
    conversation_history[:] = conversation_history[-(MAX_HISTORY_TURNS * 2):]

    # ---------------- PLAYTEST LOG ----------------
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            transcript,
            reply,
            emotion,
            confidence,
            behavior_mode,
            npc_state["trust_level"],
            list(npc_state["lore_unlocked"])
        ])

    return reply, behavior_mode
