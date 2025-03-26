import os
import numpy as np

npy_dir = "../data/ravdess_npy/Actor_01"
files = os.listdir(npy_dir)

print(f"Number of file in Actor 1 : {len(files)}")
print("Test :", files[:3])

# Try loading a .npy
mel = np.load(os.path.join(npy_dir, files[0]))
print("Spectrogram shape :", mel.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.imshow(mel, aspect='auto', origin='lower')
plt.colorbar()
plt.title("Mel spectrogram")
plt.show()
