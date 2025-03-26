import os
import librosa
import numpy as np
from glob import glob
from tqdm import tqdm

# Répertoire actuel
this_dir = os.path.abspath(os.path.dirname(__file__))
os.environ["KAGGLE_CONFIG_DIR"] = this_dir

from kaggle.api.kaggle_api_extended import KaggleApi

# === DEBUG KAGGLE CONFIG ===
print("Looking for kaggle.json in:", os.environ["KAGGLE_CONFIG_DIR"])
print("Files in directory:", os.listdir(this_dir))

# Auth Kaggle
api = KaggleApi()
api.authenticate()

# === DOWNLOAD DATASET ===
print("⏳ Downloading RAVDESS dataset...")
api.dataset_download_files(
    "uwrfkaggler/ravdess-emotional-speech-audio",
    path=this_dir,
    unzip=True
)
print("Download complete!")

# === LOCATE AUDIO FOLDER ===
print("\nSearching for audio_speech_actors_01-24 folder...")
found = False
for root, dirs, _ in os.walk(this_dir):
    for d in dirs:
        if d == "audio_speech_actors_01-24":
            wav_root = os.path.join(root, d)
            found = True
            break
    if found:
        break

if not found:
    raise FileNotFoundError("Dossier 'audio_speech_actors_01-24' non trouvé après extraction.")

print("Fichiers audio trouvés dans :", wav_root)

# === PREPARE OUTPUT DIRECTORY ===
npy_root = os.path.join(this_dir, "ravdess_npy_fixed")
os.makedirs(npy_root, exist_ok=True)

# === AUDIO PROCESSING SETTINGS ===
SR = 16000
N_MELS = 128
HOP_LENGTH = 512
TARGET_SHAPE = (128, 128)

# === FONCTION DE REDIMENSIONNEMENT ===
def resize_spectrogram(spec, target_shape):
    h, w = spec.shape
    th, tw = target_shape
    if w < tw:
        pad_width = tw - w
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=-80)
    elif w > tw:
        spec = spec[:, :tw]
    return spec

# === TRAITEMENT GLOBAL ===
actor_folders = sorted([f for f in os.listdir(wav_root) if f.startswith("Actor_")])
for actor in tqdm(actor_folders, desc="🎧 Processing and resizing .wav files"):
    actor_wav_path = os.path.join(wav_root, actor)
    actor_npy_path = os.path.join(npy_root, actor)
    os.makedirs(actor_npy_path, exist_ok=True)

    wav_files = glob(os.path.join(actor_wav_path, "*.wav"))

    for wav_file in wav_files:
        filename = os.path.basename(wav_file)
        npy_filename = filename.replace(".wav", ".npy")
        npy_file_path = os.path.join(actor_npy_path, npy_filename)

        try:
            y, sr = librosa.load(wav_file, sr=SR)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_fixed = resize_spectrogram(mel_db, TARGET_SHAPE)
            np.save(npy_file_path, mel_fixed)
        except Exception as e:
            print(f"❗ Error processing {wav_file}: {e}")

print(f"\nTous les fichiers .wav ont été convertis et redimensionnés à {TARGET_SHAPE} dans '{npy_root}'")

# === SUPPRESSION DES DOSSIERS ACTOR_XX REDONDANTS ===
print("\nSuppression des dossiers Actor_XX hors de 'audio_speech_actors_01-24'...")
deleted = 0

for item in os.listdir(this_dir):
    item_path = os.path.join(this_dir, item)
    if os.path.isdir(item_path) and item.startswith("Actor_"):
        try:
            import shutil
            shutil.rmtree(item_path)
            deleted += 1
            print(f"Supprimé : {item}")
        except Exception as e:
            print(f"Erreur lors de la suppression de {item}: {e}")

if deleted == 0:
    print("Aucun dossier 'Actor_XX' à supprimer.")
else:
    print(f"{deleted} dossier(s) 'Actor_XX' supprimé(s).")
