import torch
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import contextlib
import io

from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image

from src.data_loader import get_loaders    
from src.model import run_model 

from .config import SAVE_MODEL_DIR

# Load model
def load_best_model():

    with contextlib.redirect_stdout(io.StringIO()):
        model, optimizer, criterion, device = run_model()
        model_state = torch.load(SAVE_MODEL_DIR, map_location=device)
        model, optimizer, criterion, device = run_model()

    model.load_state_dict(model_state)
    model.eval()

    return model, device

# Evaluasi
def evaluate(model, val_loader, class_names, device):

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="  Evaluasi"):
            preds = model(images.to(device)).argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"\n  Accuracy   : {accuracy*100:.2f}%")
    print(f"  Macro F1   : {f1*100:.2f}%")

    # Classification report
    report_str = classification_report(all_labels, all_preds, target_names=class_names)
    print(f"\n  Classification Report:\n{report_str}")

    # Confusion matrix
    cm_array = confusion_matrix(all_labels, all_preds)
    fig, ax  = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm_array, display_labels=class_names).plot(
        ax=ax, colorbar=True, xticks_rotation=45
    )
    ax.set_title("Confusion Matrix — MobileNetV3-Small")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Log ke MLflow
    if mlflow.active_run():
        mlflow.log_metrics({"eval_accuracy": accuracy, "eval_macro_f1": f1})
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_text(report_str, "classification_report.txt")

    return report_str, cm_array

def run_evaluation():
    with contextlib.redirect_stdout(io.StringIO()):
        train_loader, val_loader, train_ds, val_ds, class_names, class_weights = get_loaders()

    model, device = load_best_model()
    evaluate(model, val_loader, class_names, device)

    # gradcam(model, target_layer, val_loader, class_names, device, num_images=num_gradcam)