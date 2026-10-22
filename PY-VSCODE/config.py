"""Shared settings for the Traffic Sign Recognition project (PRAICP-1002)."""
from pathlib import Path

ROOT        = Path(__file__).resolve().parent
DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"
ASSETS_DIR  = ROOT / "assets"
REF_DIR     = ASSETS_DIR / "reference"     # one reference photo per sign (used by the web app)
SAMPLES_DIR = ASSETS_DIR / "samples"       # one unseen test photo per sign (demo images in the web app)
REPORT_DIR  = ASSETS_DIR / "reports"       # confusion matrix, training curves, classification report

MODEL_PATH   = MODELS_DIR / "TrafficSign_CNN_Model.keras"
CLASS_JSON   = MODELS_DIR / "TrafficSign_class_names.json"
METRICS_JSON = MODELS_DIR / "metrics.json"

DATASET_URL = "https://d3ilbtxij3aepc.cloudfront.net/projects/AI-Capstone-Projects/PRAICP-1002-TrafSignDetc.zip"
ZIP_PATH    = DATA_DIR / "PRAICP-1002-TrafSignDetc.zip"

IMG_SIZE = 64
SEED     = 42
IMG_EXT  = (".jpg", ".jpeg", ".png", ".ppm", ".bmp")

# Hyperparameters found by the Keras Tuner search in the notebook (used with --skip-tuning)
DEFAULT_BEST_HP = {"filters_1": 32, "filters_2": 128, "dense_units": 128,
                   "dropout": 0.4, "learning_rate": 0.001, "batch_size": 16}

# Readable names - folders 0-42 follow the standard 43-class (GTSRB) numbering
GTSRB_NAMES = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)", 3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)", 6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)", 9: "No passing", 10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at next intersection", 12: "Priority road", 13: "Yield", 14: "Stop", 15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited", 17: "No entry", 18: "General caution", 19: "Dangerous curve to the left",
    20: "Dangerous curve to the right", 21: "Double curve", 22: "Bumpy road", 23: "Slippery road",
    24: "Road narrows on the right", 25: "Road work", 26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End of all speed and passing limits", 33: "Turn right ahead", 34: "Turn left ahead", 35: "Ahead only",
    36: "Go straight or right", 37: "Go straight or left", 38: "Keep right", 39: "Keep left",
    40: "Roundabout mandatory", 41: "End of no passing", 42: "End of no passing by vehicles over 3.5 tons"}
