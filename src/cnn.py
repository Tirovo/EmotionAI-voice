import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(EmotionCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16) # Stabilizes and accelerates training by normalizing activations (prevents internal covariate shift, which happens when there's too many variations between layers, for example)
        self.pool1 = nn.MaxPool2d(2)  # Reduces spatial dimensions while keeping important features (128 -> 64)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)  # (64 -> 32) 

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)  # (32 -> 16)

        self.dropout = nn.Dropout(0.6) # Regularization method to prevent overfitting (During training, randomly deactive 60% of all neurons)

        self.fc1 = nn.Linear(64 * 16 * 16, 128) #  Linear : Performs classification based on extracted features
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (B, 16, 64, 64) relu : Adds non-linearity, enabling the model to learn complex patterns
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # (B, 32, 32, 32)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # (B, 64, 16, 16)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)  # Flatten : Converts 3D feature maps into 1D vectors
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


model = EmotionCNN()
sample_input = torch.randn(1, 1, 128, 128)
out = model(sample_input)
print(out.shape)
