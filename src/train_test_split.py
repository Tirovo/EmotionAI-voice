from dataset import EmotionDataset
from torch.utils.data import DataLoader

# Split actors in two groups : one group for training and one to test the model
train_actors = list(range(1, 21))
test_actors = list(range(21, 25))

# Dataset
train_set = EmotionDataset("../data/ravdess_npy_fixed", actors=train_actors)
test_set   = EmotionDataset("../data/ravdess_npy_fixed", actors=test_actors)

# DataLoader
train_loader = DataLoader(train_set, batch_size=32, shuffle=True) # Load 32 samples per batch and shuffle data to prevent order bias
test_loader   = DataLoader(test_set, batch_size=32)

print(train_actors, test_actors)
