from datetime import datetime

# path direktori
DATA_DIR_RAW        = "data/raw"
DATA_DIR_SPLIT      = "data/processed"
DATA_DIR_PROCESSED  = "data/processed/training"
TRAINING_PLOT_DIR = "img/training_curve.png"
SAVE_MODEL_DIR = "models/best_model.pt"

# build & training model
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 10
FREEZE_UNTIL = 8  

# MlFlow
MLFLOW_NAME = "Tomato_Disease_MobileNetV3"
RUN_NAME = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
