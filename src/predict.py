import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

from .config  import SAVE_MODEL_DIR, DATA_DIR_SPLIT
from .model   import run_model

import contextlib, io

# load model
def load_model():
    with contextlib.redirect_stdout(io.StringIO()):
        model, _, _, device = run_model()
    model.load_state_dict(torch.load(SAVE_MODEL_DIR, map_location=device))
    model.eval()
    return model, device

# load data validasi
def load_val_data(num_samples: int = 10):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_ds    = datasets.ImageFolder(f"{DATA_DIR_SPLIT}/val", transform=transform)
    loader    = DataLoader(val_ds, batch_size=num_samples, shuffle=True)
    images, labels = next(iter(loader))
    return images, labels, val_ds.classes

# prediksi
def predict(model, images, device):
    with torch.no_grad():
        outputs = model(images.to(device))
        probs   = torch.softmax(outputs, dim=1)
        preds   = probs.argmax(dim=1).cpu()
        confs   = probs.max(dim=1).values.cpu()
    return preds, confs

# denormalisasi
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

# visualisasi 
def plot_predictions(images, labels, preds, confs, class_names,
                     ncols: int = 5, save_path: str = "img/prediction_grid.png"):

    nrows  = len(images) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.4))
    axes   = axes.flatten()

    # Label pendek (nama kelas tanpa prefix "Tomato_")
    short = [c.replace("Tomato_", "").replace("Tomato__", "")
               .replace("_", " ") for c in class_names]

    for i, ax in enumerate(axes):
        img = denormalize(images[i])
        ax.imshow(img)
        ax.axis("off")

        actual = short[labels[i]]
        pred   = short[preds[i]]
        conf   = confs[i].item() * 100
        correct = labels[i] == preds[i]

        color  = "#99AD7A" if correct else "#D96868"   # hijau / merah
        mark   = "true" if correct else "false"

        ax.set_title(
            f"{mark} Pred : {pred}\n({conf:.1f}%)\nAktual: {actual}",
            fontsize=8, color=color, fontweight="bold", pad=4
        )

    # legend
    patches = [
        mpatches.Patch(color="#2ecc71", label="Prediksi Benar"),
        mpatches.Patch(color="#e74c3c", label="Prediksi Salah"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

    correct_total = (labels == preds).sum().item()
    fig.suptitle(
        f"Perbandingan Aktual vs Prediksi  |  "
        f"Benar: {correct_total}/{len(images)}  ({correct_total/len(images)*100:.1f}%)",
        fontsize=13, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Grid disimpan : {save_path}")

# entry point
def run_predict(num_samples: int = 10):
    print("\n  [PREDICT] Loading model")
    model, device = load_model()

    print("  [PREDICT] Loading validation data")
    images, labels, class_names = load_val_data(num_samples)

    print("  [PREDICT] Running prediction")
    preds, confs = predict(model, images, device)

    print("  [PREDICT] Plotting results")
    plot_predictions(images, labels, preds, confs, class_names)

    return images, labels, preds, confs, class_names