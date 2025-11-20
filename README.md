# Emotional NPC

**A Real-Time Emotion-Aware NPC System for Unreal Engine**

[![Unreal Engine](https://img.shields.io/badge/Unreal%20Engine-5.x-blue)](https://www.unrealengine.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Emotional NPC is a real-time interactive character system that reacts to the player's facial expressions, spoken language, and soon, tone of voice. The system uses multimodal emotion sensing (FER + STT + LLM + TTS) to create NPC conversations that feel more natural, responsive, and narratively meaningful.

Designed as a modular framework, this project will evolve into a drop-in emotional AI toolkit for Unreal Engine.

## Why This Project Matters

Traditional NPCs rely on fixed dialogue trees that ignore the player's emotional state. This research explores whether emotionally responsive NPCs can:

- **Improve immersion** in interactive experiences
- **Increase narrative engagement** through adaptive storytelling
- **Support more human-like interaction** that feels natural
- **Encourage empathy-driven player decisions** based on emotional connection

The system also advances research in affective computing and interactive narrative design.

## Research Question

> *How does real-time emotional awareness—through facial expression, vocal cues, and natural language—shape player perception of NPC empathy, presence, and narrative engagement?*

## User Research & Early Feedback

Across **30+ responses**, players consistently described emotion-responsive NPCs as more immersive and meaningful. However, realism only improves when emotional responses:

- Support player agency
- Avoid overreaction
- Do not break roleplay

These insights shaped the system's design: subtle emotional cues, short responses, and conversational memory.

## System Features

### 1. Real-Time Facial Emotion Recognition
- DeepFace model detecting happy, sad, and neutral states
- Confidence smoothing for stability
- Live OSC streaming into Unreal Engine

### 2. Speech-to-Text (Player Input)
- Whisper transcribes player speech
- Combined with emotion snapshot
- Sent to LLM for contextual response generation

### 3. LLM-Driven Dialogue
- GPT-4o-mini generates Commander Zad's replies
- Persona-based, emotionally adaptive responses
- Lightweight conversation memory buffer
- First sentence always reacts to player emotion

### 4. In-Engine TTS + Animation
- Unreal Engine Text-to-Speech subsystem
- NPC speaks every generated line
- Modular talking montage system (rotating 4 clips)

### 5. Unreal Engine OSC Integration
- `/npc/reply` — NPC dialogue delivery
- `/speech` — Player transcript
- `/emotion/label`, `/emotion/confidence` — Facial emotion recognition
- Drives UI, TTS, animation, and reaction logic

## Research Methods & Evaluation Strategy

### A. Experimental Design
- **Between-subjects study**
  - Group A: Emotionally responsive NPC
  - Group B: Static/scripted NPC
- Compare empathy, immersion, and narrative engagement

### B. Quantitative Instruments
- **Game Experience Questionnaire (GEQ)**: immersion, flow, enjoyment
- **ITC-Sense of Presence Inventory (ITC-SOPI)**: spatial + social presence
- **Adapted Empathy Scale**: emotional connection with NPC
- **Emotion Logs**: facial and vocal affect before and after interaction

### C. Qualitative Evaluation
- Post-play interviews
- Anonymous follow-up survey
- Annotated gameplay footage for emotional events
- Open-ended questions on NPC realism and emotional response

### D. Metrics Tracked
- Response time to emotionally charged lines
- Number of dialogue branches explored
- Emotional shifts during interaction
- Changes in affect over time
- Memory and past emotions

### Rationale
- Grounded in affective computing and HCI literature
- Multimodal sensing improves accuracy
- Mixed-methods provide depth and generalizability

## Impact on Player Experience and Games

### Benefits to Game Design
- NPCs feel more alive and responsive
- Supports narrative-driven decision-making
- Encourages player empathy and connection
- Makes dialogue adaptable to emotional context

### Broader Applications
- Education and training simulations
- AI companions that respond to user frustration or confusion
- Interactive storytelling systems

## Future Features

- [ ] Emotional decay system (NPC gradually resets mood)
- [ ] Emotion-linked quests
- [ ] Cross-modal sarcasm or contradiction detection
- [ ] Valence–Arousal continuous emotion mapping
- [ ] Full SER (Speech Emotion Recognition) integration

---

**Star ⭐ this repository if you find it useful!**
