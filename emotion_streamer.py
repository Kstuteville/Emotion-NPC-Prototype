"""
emotion_stream.py
Author: Kaylie Stuteville
Project: Emotion-Responsive NPC Villager Prototype

Description:
This script performs real-time facial emotion detection using DeepFace,
applies exponential smoothing to reduce temporal volatility, and streams
the dominant smoothed emotion to both Unreal Engine (via OSC) and an LLM
interface (via WebSocket).

Purpose:
The goal is to provide a continuous, human-like emotional signal based
on visual cues. The smoothed emotion values allow for stable, gradual
NPC behavioral responses (e.g., animation, lighting) and tone modulation
in AI dialogue systems.

Method Overview:
1. Capture webcam frames.
2. Detect emotion probabilities using DeepFace.
3. Apply exponential smoothing across time:
       S_t = α * x_t + (1 - α) * S_{t-1}
   where α is the smoothing factor (0.2 = slow, natural change).
4. Send the dominant emotion and confidence values at a fixed interval
   (e.g., every 2 seconds) via:
       a) OSC to Unreal Engine → drives animations and lighting.
       b) WebSocket → provides emotion context to LLM dialogue system.

Dependencies:
    pip install deepface python-osc websockets opencv-python
"""

from deepface import DeepFace
from pythonosc import udp_client
import asyncio, websockets, cv2, time, json
from datetime import datetime
from collections import defaultdict

# -------------------- CONFIGURATION --------------------

OSC_IP = "127.0.0.1"       # Unreal Engine OSC receiver address
OSC_PORT = 9000            # Unreal OSC port (configure in Unreal)
WS_PORT = 8765             # WebSocket port for LLM listener
ALPHA = 0.2                # Smoothing factor (0.2 = stable, human-like)
INTERVAL = 2.0             # Time between emotion updates (seconds)

# Initialize OSC client for Unreal Engine
osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)


# -------------------- SMOOTHING FUNCTION --------------------

def smooth(prev_scores, new_scores, alpha=ALPHA):
    """
    Apply exponential smoothing to emotion confidence scores.
    prev_scores: dict of previous smoothed values
    new_scores:  dict of current frame's raw emotion confidences
    alpha:       smoothing factor (0–1)
    """
    return {k: alpha * float(new_scores.get(k, 0))
            + (1 - alpha) * prev_scores.get(k, 0)
            for k in set(prev_scores) | set(new_scores)}


# -------------------- MAIN EMOTION LOOP --------------------

async def emotion_loop(websocket=None):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    smoothed = defaultdict(float)
    last_sent = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        current_time = time.time()

        # Perform analysis at defined intervals only
        if current_time - last_sent >= INTERVAL:
            result = DeepFace.analyze(
                frame, actions=["emotion"],
                enforce_detection=False, silent=True
            )[0]

            # Apply exponential smoothing
            smoothed = smooth(smoothed, result["emotion"])
            dominant = max(smoothed, key=smoothed.get)
            confidence = round(float(smoothed[dominant]), 2)
            timestamp = datetime.now().strftime("%H:%M:%S")

            # Send data to Unreal Engine via OSC
            osc_client.send_message("/emotion", [dominant, confidence])

            # Send data to WebSocket client (e.g., LLM tone modulator)
            if websocket:
                msg = json.dumps({
                    "emotion": dominant,
                    "confidence": confidence,
                    "timestamp": timestamp
                })
                await websocket.send(msg)

            last_sent = current_time

        await asyncio.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()


# -------------------- WEBSOCKET SERVER --------------------

async def ws_handler(websocket):
    """Handles WebSocket connections and streams emotion data."""
    await emotion_loop(websocket)


async def main():
    """Starts WebSocket server for concurrent OSC and WebSocket streaming."""
    async with websockets.serve(ws_handler, "localhost", WS_PORT):
        await asyncio.Future()  # Keeps the server running indefinitely


# -------------------- ENTRY POINT --------------------

if __name__ == "__main__":
    asyncio.run(main())
