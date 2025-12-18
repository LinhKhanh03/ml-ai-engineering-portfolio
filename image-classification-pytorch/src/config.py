from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "dataset"
PREDICT_DIR = PROJECT_ROOT / "inference_images"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
CLASSES_PATH = PROJECT_ROOT / "classes.txt"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
EPOCHS = 5
