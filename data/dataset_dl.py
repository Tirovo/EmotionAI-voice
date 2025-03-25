import os

# 🔐 Set the Kaggle config dir BEFORE importing the Kaggle API
this_dir = os.path.abspath(os.path.dirname(__file__))
os.environ["KAGGLE_CONFIG_DIR"] = this_dir

from kaggle.api.kaggle_api_extended import KaggleApi  # imported AFTER setting env var

# Debug
print("Looking for kaggle.json in:", os.environ["KAGGLE_CONFIG_DIR"])
print("Files:", os.listdir(this_dir))

# Auth + download
api = KaggleApi()
api.authenticate()

print("⏳ Downloading RAVDESS dataset...")
api.dataset_download_files(
    "uwrfkaggler/ravdess-emotional-speech-audio",
    path=this_dir,
    unzip=True
)

print("✅ Done. Files are in:", this_dir)

