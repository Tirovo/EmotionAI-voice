# 🎙️ EmotionAI Voice

> An AI-powered application for detecting human emotions through voice input, designed for healthcare and human-computer interaction enhancement.

---

## 📌 Overview

**EmotionAI Voice** is an open-source deep learning project that classifies vocal emotions using raw `.wav` audio.  
It's designed for applications in mental health monitoring, UX analysis, and intelligent speech interfaces.

🔬 The model is trained **from scratch**, using spectrogram-based audio features, and aims to recognize 8 core emotions.

---

## 🎯 Features

- 🧠 Emotion recognition: `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`
- 🎧 Accepts `.wav` audio inputs (from RAVDESS dataset)
- 📊 CNN and CNN+GRU models implemented in PyTorch
- 🔍 Real-time evaluation with confusion matrix and accuracy tracking
- 🛠️ Fully open-source and customizable (no pre-trained models)
- 🧪 Includes **SpecAugment** for data augmentation (frequency/time masking)

---

## 📚 Dataset — RAVDESS

We use the [**RAVDESS** dataset](https://zenodo.org/record/1188976), which includes:

- 🎭 24 professional actors (balanced male/female)
- 🎙️ 1440 `.wav` files (16-bit, 48kHz)
- 8 labeled emotions:  
  `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`

Each `.wav` file is preprocessed into a **Mel spectrogram** and stored as `.npy` format.

---

## 🧠 Model Architectures

### 2 different models
#### ✅ CNN (Best Performance)
- 3x Conv1D + ReLU + MaxPool
- Fully connected layers
- Dropout regularization (adjustable)

#### 🔁 CNN + GRU
- CNN front-end for spatial encoding
- GRU (recurrent layers) to capture temporal dynamics
- Lower accuracy than CNN-only model

---

### 🧪 SpecAugment: Data Augmentation

To improve generalization, we implemented `SpecAugmentTransform` which applies:

- 🕒 **Time masking**: hides random time intervals
- 📡 **Frequency masking**: hides random mel frequency bands

---

### 📈 Training Results

- Best Validation Accuracy: **~49.6%**
- Training set: Actors 1–20
- Validation set: Actors 21–24

---

**Confusion Matrix Example:**

![Confusion Matrix](./confusion_matrix.png)

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt

### 2. Download dataset from Kaggle

Follow the instructions in the README.md located in the data folder

### 3 . Train the model

```bash
python src/train.py

### 4.  Evaluation the performances with a confusion matrix

```bash
python src/confusion_matrix.py


