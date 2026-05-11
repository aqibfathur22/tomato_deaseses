import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.manifold import TSNE

from .config  import SAVE_MODEL_DIR, DATA_DIR_SPLIT
from .model   import run_model

import contextlib, io, random

# denormalisasi
def _denorm(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

# load model
def _load(device=None):
    with contextlib.redirect_stdout(io.StringIO()):
        model, _, _, device = run_model()
    model.load_state_dict(torch.load(SAVE_MODEL_DIR, map_location=device))
    model.eval()
    return model, device

# load data validasi
def _val_loader(batch_size=64, shuffle=True):
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(f"{DATA_DIR_SPLIT}/val", transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle), ds.classes

# 1. visualsisasi grad-cam
class GradCAM:

    def __init__(self, model, target_layer):
        self.model      = model
        self.features   = None
        self.gradients  = None

        target_layer.register_forward_hook(
            lambda m, inp, out: setattr(self, 'features', out.detach())
        )
        target_layer.register_full_backward_hook(
            lambda m, grad_in, grad_out: setattr(self, 'gradients', grad_out[0].detach())
        )

    def __call__(self, image_tensor, class_idx=None):
        image_tensor = image_tensor.unsqueeze(0).requires_grad_(True)

        logits  = self.model(image_tensor)
        target  = logits[0, class_idx if class_idx is not None else logits.argmax()]

        self.model.zero_grad()
        target.backward()

        weights = self.gradients.mean(dim=[0, 2, 3])

        cam = (weights[:, None, None] * self.features[0]).sum(dim=0)
        cam = F.relu(cam)

        cam -= cam.min(); cam /= cam.max() + 1e-8
        cam  = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze().numpy()

        return cam

def run_gradcam(num_images: int = 4,
                save_path: str = "img/gradcam.png"):

    model, device = _load()
    loader, class_names = _val_loader(batch_size=num_images)

    gradcam = GradCAM(model, target_layer=model.features[-1])
    short   = [c.replace("Tomato_", "").replace("Tomato__", "")
                .replace("_", " ") for c in class_names]

    images, labels = next(iter(loader))
    images, labels = images[:num_images], labels[:num_images]

    fig, axes = plt.subplots(num_images, 2,
                             figsize=(6, num_images * 2.8))

    for i in range(num_images):
        img_tensor = images[i].to(device)
        img_rgb    = _denorm(images[i])
        label      = labels[i].item()

        with torch.enable_grad():
            cam = gradcam(img_tensor, class_idx=label)

        # Heatmap overlay
        heatmap = cm.jet(cam)[..., :3]             
        overlay = (0.55 * img_rgb + 0.45 * heatmap).clip(0, 1)

        axes[i, 0].imshow(img_rgb);  axes[i, 0].axis("off")
        axes[i, 1].imshow(overlay);  axes[i, 1].axis("off")

        if i == 0:
            axes[i, 0].set_title("Gambar Asli",    fontsize=10, fontweight="bold")
            axes[i, 1].set_title("Grad-CAM Overlay", fontsize=10, fontweight="bold")

        axes[i, 0].set_ylabel(short[label], fontsize=7, rotation=0,
                              labelpad=60, va="center")

    fig.suptitle("Grad-CAM — Area Fokus Model per Kelas Penyakit",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Grad-CAM disimpan : {save_path}")

# 2. visualisasi t-SNE = Distribusi Kelas di Feature Space
def _extract_features(model, loader, device, max_batches=10):

    features_list, labels_list = [], []

    pool_out = {}
    hook = model.avgpool.register_forward_hook(
        lambda m, inp, out: pool_out.update({'feat': out.detach()})
    )

    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(loader):
            if i >= max_batches:
                break
            model(imgs.to(device))
            feat = pool_out['feat'].squeeze(-1).squeeze(-1).cpu().numpy()
            features_list.append(feat)
            labels_list.extend(lbls.numpy())

    hook.remove()
    return np.concatenate(features_list), np.array(labels_list)


def run_tsne(max_samples: int = 500,
             save_path:   str = "img/tsne.png"):

    model, device = _load()
    loader, class_names = _val_loader(batch_size=64, shuffle=True)
    short = [c.replace("Tomato_", "").replace("Tomato__", "")
              .replace("_", " ") for c in class_names]

    print("  [t-SNE] Extracting features...")
    feats, labels = _extract_features(model, loader, device,
                                       max_batches=max_samples // 64 + 1)

    idx    = random.sample(range(len(feats)), min(max_samples, len(feats)))
    feats  = feats[idx]
    labels = labels[idx]

    print("  [t-SNE] Running dimensionality reduction...")
    emb = TSNE(n_components=2, perplexity=40, max_iter=1000,
               random_state=42).fit_transform(feats)

    # Plot
    colors = plt.cm.get_cmap("tab10", len(class_names))
    fig, ax = plt.subplots(figsize=(10, 8))

    for cls_id, cls_name in enumerate(short):
        mask = labels == cls_id
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=[colors(cls_id)], label=cls_name,
                   alpha=0.7, s=18, edgecolors="none")

    ax.set_title("t-SNE — Distribusi Kelas di Feature Space\n"
                 "(Cluster terpisah = representasi fitur baik)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE Dim 1"); ax.set_ylabel("t-SNE Dim 2")
    ax.legend(fontsize=8, markerscale=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  t-SNE disimpan : {save_path}")

# entry point
def run_xai(num_gradcam: int = 4, num_tsne: int = 500):
    print("\n  ── Grad-CAM ─────────────────────────────")
    run_gradcam(num_images=num_gradcam)

    print("\n  ── t-SNE ────────────────────────────────")
    run_tsne(max_samples=num_tsne)