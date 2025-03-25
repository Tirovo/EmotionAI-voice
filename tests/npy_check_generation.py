import os
import numpy as np

npy_dir = "../data/ravdess_npy/Actor_01"
files = os.listdir(npy_dir)

print(f"🧠 Nombre de fichiers dans Actor_01 : {len(files)}")
print("🔎 Exemple :", files[:3])

# Charger un .npy pour voir
mel = np.load(os.path.join(npy_dir, files[0]))
print("✅ Shape du spectrogramme :", mel.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.imshow(mel, aspect='auto', origin='lower')
plt.colorbar()
plt.title("Mel spectrogram")
plt.show()