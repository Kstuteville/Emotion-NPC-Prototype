import cv2
import random
import numpy as np
from state import npc_state
from fer import detect_emotion
from voice import transcribe
from npc_brain import generate_npc_reply
from ace_schema import build_ace_payload
from renderer_unreal import render

last_emotion = "neutral"
last_conf = 1.0
last_should_comment = False

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    emotion, conf = detect_emotion(frame)
    last_emotion, last_conf = emotion, conf
    last_should_comment = random.random() < 0.3

    # PTT + audio logic remains exactly as before
    # transcript = transcribe(audio_np)

    emotion_packet = {
        "label": emotion,
        "confidence": conf,
        "should_comment": last_should_comment
    }

    reply, behavior_mode = generate_npc_reply(transcribe, emotion_packet)

    ace_payload = build_ace_payload(
        reply, emotion, conf, behavior_mode, npc_state
    )

    render(ace_payload)
