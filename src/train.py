import time
import copy
import torch
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt

from tqdm import tqdm

from .config import EPOCHS, LEARNING_RATE, BATCH_SIZE, TRAINING_PLOT_DIR, SAVE_MODEL_DIR, MLFLOW_NAME, RUN_NAME
from .data_loader import get_loaders     
from .model import run_model      

# training model
def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):

    model.train() if is_train else model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, desc="  Train" if is_train else "  Val  ", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

# visualisasi
def plot_curves(history: dict):

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(history["val_loss"],   label="Val Loss",   marker="o")
    axes[0].set_title("Loss per Epoch"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(history["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(history["val_acc"],   label="Val Acc",   marker="o")
    axes[1].set_title("Accuracy per Epoch"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(TRAINING_PLOT_DIR, dpi=150)
    plt.show()
    print(f"  Kurva disimpan : {TRAINING_PLOT_DIR}")
    return TRAINING_PLOT_DIR

def run_training(run_name: str = "run_01",):
    # 1. prepare data
    train_loader, val_loader, train_ds, val_ds, class_names, class_weights = get_loaders()
    # 2. Build model
    model, optimizer, criterion, device= run_model(class_weights=class_weights)
    # 3. Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    # 4. Ml flow
    mlflow.set_experiment(MLFLOW_NAME)

    with mlflow.start_run(run_name=RUN_NAME):

        mlflow.log_params({
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": type(optimizer).__name__,
            "scheduler": "CosineAnnealingLR",
            "num_classes": len(class_names),
            "model": "MobileNetV3-Small",
            "freeze_until": 8,
            "dropout": 0.4,
            # "label_smoothing": 0.15,
        })

        # 5. Loop epoch 
        history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_acc     = 0.0
        best_weights = copy.deepcopy(model.state_dict())
        patience, no_improve = 3, 0
        start_time   = time.time()

        for epoch in range(1, EPOCHS + 1):
            print(f"\n  Epoch {epoch}/{EPOCHS}  LR: {scheduler.get_last_lr()[0]:.6f}")

            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, is_train=False)
            scheduler.step()

            # simpan history & print
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            print(f"  Train  ->  Loss : {train_loss:.4f}  |  Accuracy : {train_acc*100:.2f}%")
            print(f"  Val    ->  Loss : {val_loss:.4f}  |  Accuracy : {val_acc*100:.2f}%")

            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss,   "val_acc": val_acc,
            }, step=epoch)

            # simpan best epoch
            if val_acc > best_acc:
                best_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(best_weights, SAVE_MODEL_DIR)
                print(f"  Model terbaik disimpan (val_acc : {best_acc*100:.2f}%)")
                no_improve = 0
            else:
                no_improve += 1

            # Early stopping
            if no_improve >= patience :
                print(f"\n  Early stopping : {patience} epoch tanpa peningkatan.")
                break

        # 6. pasca training 
        elapsed = time.time() - start_time
        print(f"\n  Training Done")
        print("  ------------------------")
        print(f"  Duration : {elapsed/60:.2f} minutes")
        print(f"  Best Val Accuracy : {best_acc*100:.2f}%")

        model.load_state_dict(best_weights)

        mlflow.log_artifact(plot_curves(history))
        mlflow.pytorch.log_model(model, artifact_path="model")
        mlflow.log_artifact(SAVE_MODEL_DIR, artifact_path="checkpoints")
        mlflow.log_metric("best_val_acc", best_acc)

        print(f"\n  MLflow run selesai")