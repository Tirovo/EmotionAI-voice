import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob

# Mapping émotion ID → label (tu peux adapter)
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

class EmotionDataset(Dataset):
    def __init__(self, root_dir, actors, transform=None, normalize=True):
        self.files = []
        self.labels = []
        self.transform = transform
        self.normalize = normalize

        for actor_id in actors:
            actor = f"Actor_{str(actor_id).zfill(2)}"
            actor_path = os.path.join(root_dir, actor)
            npy_files = glob(os.path.join(actor_path, "*.npy"))

            for file in npy_files:
                self.files.append(file)
                label_code = os.path.basename(file).split("-")[2]
                self.labels.append(int(label_code) - 1)  # convert to 0-based index

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spec = np.load(self.files[idx])

        # Normalisation [0, 1] simple (optionnelle)
        if self.normalize:
            spec = (spec + 80) / 80  # car valeurs entre [-80, 0]

        # (1, 128, 128) pour CNN 2D
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            spec = self.transform(spec)

        return spec, label