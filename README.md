# Emotional NPC

**A Real-Time Emotion-Aware NPC System for Unreal Engine**

An interactive NPC system that responds to player facial expressions, speech content, and emotional context in real time — creating dialogue that feels alive, adaptive, and narratively meaningful.

---

## Overview

Traditional NPCs rely on rigid dialogue trees that ignore how the player actually feels.  
**Emotional NPC** changes this by giving characters the ability to read and respond to real emotional signals through multimodal AI sensing.

Meet **Commander Zad** — a character who adapts their tone, reveals lore gradually, assigns quests, and builds trust based on your facial expressions, voice, and conversational history.

This isn’t just cosmetic: emotion shapes the narrative, unlocks quests, and influences relationship progression.

Designed as a modular framework, this project will evolve into a drop-in emotional AI toolkit for Unreal Engine.

---

## Why This Project Matters

Traditional NPCs rely on fixed dialogue trees that ignore the player's emotional state.  
This research explores whether emotionally responsive NPCs can:

- **Improve immersion** in interactive experiences  
- **Increase narrative engagement** through adaptive storytelling  
- **Support more human-like interaction** that feels natural  
- **Encourage empathy-driven player decisions** based on emotional connection  

The system also advances research in affective computing and interactive narrative design.

---

## Research Question

> *How does real-time emotional awareness — through facial expression and natural language — affect player immersion, agency, and narrative engagement?*

This prototype explores whether emotionally responsive NPCs can create deeper connections without breaking player control.

---

## User Research & Early Feedback

Across **30+ playtests and surveys**:

- **"Felt more immersive"** — dialogue reactions seemed natural and responsive  
- **"More meaningful conversations"** — emotional alignment created stronger NPC attachment  
- **"Like they actually listened"** — subtle emotion reads increased presence  

But realism only worked when:

- Emotional responses stayed **subtle** (not overwhelming)  
- Player agency remained **intact** (no hijacking conversation)  
- Reactions matched **roleplay context** (not breaking immersion)  

These insights shaped core design decisions: **emotion throttling**, **trust-gated lore**, and **agency-first dialogue**.

---

## Core Features

### **Multimodal Emotion Sensing**
- Facial Expression Recognition — DeepFace detects happy, sad, neutral states  
- Speech-to-Text — Whisper transcribes player dialogue  
- Tone Analysis — *Coming soon* (Speech Emotion Recognition)  
- Confidence Smoothing — Prevents jittery emotion detection  

### 🧠 Adaptive AI Personality
- LLM-Driven Responses — GPT-4o-mini generates contextual dialogue  
- Emotional Memory — remembers past interactions and player mood  
- Dynamic Trust System — relationship evolves based on player behavior  
- Mood States — Commander Zad has guilt, suspicion, sympathy, and trauma states  

### 🎮 Unreal Engine Integration
- OSC Communication — real-time Python ↔ Unreal data streaming  
- Text-to-Speech — in-engine voice synthesis for NPC responses  
- Talking Animations — lip-sync and gesture system  
- UI Feedback — live emotion display and dialogue rendering  

### 📖 Emergent Narrative Systems
- Trust-Gated Lore — story reveals unlock through relationship progression  
- Emotion-Driven Quests — three adaptive storylines (Repair, Anomaly, Confession)  
- Contextual Objectives — NPC suggests tasks based on emotional state and conversation  
- Behavioral Adaptation — teasing when you smile, cautious when you're sad  

---

## Impact on Player Experience and Games

### Benefits to Game Design
- NPCs feel more alive and responsive  
- Supports narrative-driven decision-making  
- Encourages player empathy and connection  
- Makes dialogue adaptable to emotional context  

### Broader Applications
- Education and training simulations  
- AI companions that respond to frustration or confusion  
- Interactive storytelling experiences  

---

## Future Features

- [ ] Emotional decay system (NPC gradually resets mood)  
- [ ] Cross-modal sarcasm or contradiction detection  
- [ ] Custom character TTS voices
- [ ] Full SER (Speech Emotion Recognition) integration  

---

**Star ⭐ this repository if you find it useful!**
