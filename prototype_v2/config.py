import os
from pynput import keyboard

LOG_FILE = "npc_playtest_log.csv"

UE_IP = "" 
UE_PORT = 9000

SAMPLE_RATE = 16000
FER_ALPHA = 0.25
FER_FRAME_STRIDE = 5

EMOTIONS = ["neutral", "happy", "sad"]

PTT_KEY = keyboard.KeyCode.from_char('v')

MAX_HISTORY_TURNS = 6
