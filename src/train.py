import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import EmotionDataset
from cnn import EmotionCNN
from spec_augment_transform import SpecAugmentTransform

import os

# Use your GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PATHS
data_path = "../data/ravdess_npy_fixed"
save_path = "../models/emotion_cnn.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Data Loading
train_actors = list(range(1, 21))
val_actors = list(range(21, 25))

# Training set creation with data augmentation included : improves generalization, reduces overfitting, makes the model rore robust to variations in intonation, speaking rate, and background noise overall.
train_set = EmotionDataset(
    data_path,
    actors=train_actors,
    transform=SpecAugmentTransform(),
    normalize=True
)
val_set = EmotionDataset(data_path, actors=val_actors)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

# Model
model = EmotionCNN().to(device)

# Loss + optimizer
lossFunction = nn.CrossEntropyLoss() # Loss: measures how far predictions are from true emotion labels
optimizer = optim.Adam(model.parameters(), lr=1e-3) # Optimizer : updates CNN weights efficiently using adaptive gradients
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) # Scheduler: reduces learning rate over time to refine training stability

# 🏋️‍♂️ Entraînement
EPOCHS = 25
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = lossFunction(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)

    train_acc = correct / total
    train_loss /= total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            val_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total
    val_loss /= val_total

    print(f"📚 Epoch [{epoch+1}/{EPOCHS}] | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"✅ New best model saved to {save_path} (val_acc={val_acc:.4f})")

    scheduler.step()
