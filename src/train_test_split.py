from dataset import EmotionDataset
from torch.utils.data import DataLoader

# Split par acteur
train_actors = list(range(1, 21))
test_actors = list(range(21, 25))

# Dataset
train_set = EmotionDataset("../data/ravdess_npy_fixed", actors=train_actors)
test_set   = EmotionDataset("../data/ravdess_npy_fixed", actors=test_actors)

# DataLoader
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader   = DataLoader(test_set, batch_size=32)

print(train_actors, test_actors)