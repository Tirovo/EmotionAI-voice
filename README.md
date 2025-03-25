# 🎙️ EmotionAI Voice

> An AI-powered application for detecting human emotions through voice input, designed for healthcare and human-computer interaction enhancement.

## 📌 Overview

**EmotionAI Voice** is an open-source project that leverages Deep Learning and audio signal processing to classify emotions from human speech.  
This application is designed for use cases in mental health, digital well-being, and intelligent voice interfaces.

It demonstrates how raw vocal input can be processed, analyzed, and used to infer emotional states — using a custom-built deep learning model trained entirely from scratch.

---

## 🎯 Features

- 🧠 Emotion recognition (e.g., calm, angry, happy, sad, etc.)
- 🎧 Upload `.wav` audio files
- 💬 Real-time prediction display (via Streamlit interface)
- 🧪 Custom CNN model trained using PyTorch
- 📁 Lightweight and fully customizable
- ✅ No pre-trained models used — fully from scratch

---

## 📚 Dataset

We use the [**RAVDESS** dataset](https://zenodo.org/record/1188976), which contains labeled speech recordings from 24 actors, expressing 8 different emotions.

Emotion classes included:
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

You must download the dataset manually and place the `.wav` files inside the `data/` folder.  
See `data/README.md` for more instructions.

---

## 🧠 Model

The model used is a **1D Convolutional Neural Network (CNN)** built from scratch with **PyTorch**, trained on **MFCC** features extracted from each audio sample.

No pre-trained weights or transfer learning models are used — this project is fully custom and pedagogical.

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
