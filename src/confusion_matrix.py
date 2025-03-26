import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from cnn import EmotionCNN
from dataset import EmotionDataset  # adapte selon ton fichier

# Configuration
BATCH_SIZE = 32
MODEL_PATH = "../models/emotion_cnn.pt"
DATA_PATH = "../data/ravdess_npy_fixed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_actors = list(range(21, 25))
train_actors = list(range(1, 21))


# Load
model = EmotionCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Dataset
val_dataset = EmotionDataset(DATA_PATH, actors=val_actors)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Evaluation
list_preds = []
list_actuals = []

with torch.no_grad():
    for X, y in val_loader:
        X,y = X.to(DEVICE), y.to(DEVICE)
        outputs = model(X)
        preds = torch.argmax(outputs, dim=1)
        list_preds.extend(preds.cpu().numpy())
        list_actuals.extend(y.cpu().numpy())

# Accuracy evaluation
acc = accuracy_score(list_actuals, list_preds)
print(f"Validation Accuracy: {acc*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(list_actuals, list_preds)
class_names = val_dataset.classes if hasattr(val_dataset, "classes") else [str(i) for i in range(cm.shape[0])]

emotion_map = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}

class_names = [emotion_map[i] for i in range(len(emotion_map))]
os.makedirs("../figures", exist_ok=True)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (Acc: {acc*100:.2f}%)")
plt.tight_layout()

plt.savefig("../figures/confusion_matrix.png", dpi=300)
print("Confusion matrix saved as 'figures/confusion_matrix.png'")

plt.show()

