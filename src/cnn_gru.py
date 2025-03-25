import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN_GRU(nn.Module):
    def __init__(self, num_classes=8):
        super(EmotionCNN_GRU, self).__init__()

        # CNN 2D pour extraire des features du spectrogramme
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128 -> 64x64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64 -> 32x32
        )

        # 🔁 GRU plus costaud (2 layers, hidden=128, dropout=0.3)
        self.gru = nn.GRU(
            input_size=32,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.4)

        # FC final
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)             # (B, 32, 32, 32)
        x = x.mean(dim=2)           # moyenne freq -> (B, 32, 32)
        x = x.permute(0, 2, 1)      # (B, time, features)

        _, h = self.gru(x)          # h: (num_layers*2, B, 128)
        h_last = torch.cat([h[-2], h[-1]], dim=1)  # (B, 256)

        h_last = self.dropout(h_last)
        return self.fc(h_last)

    


if __name__ == "__main__":
    model = EmotionCNN_GRU()
    x = torch.randn(8, 1, 128, 128)
    out = model(x)
    print(out.shape)  # 🔥 torch.Size([8, 8])
