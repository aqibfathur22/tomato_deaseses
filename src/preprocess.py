import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import WeightedRandomSampler

from .config import DATA_DIR_RAW, DATA_DIR_SPLIT


def clean_dataset(df, duplicates, corrupt_files):

    # 1. Hapus file corrupt dari folder split (train & val) 
    corrupt_set = set()
    for item in corrupt_files:
        rel_path = os.path.relpath(item['path'], DATA_DIR_RAW)
        parts = rel_path.split(os.sep)
        if len(parts) >= 2:
            cls, filename = parts[0], parts[1]
            corrupt_set.add((cls, filename))

    # 2. Kumpulkan file duplikat 
    duplicate_set = set()
    for dup in duplicates:
        rel_path = os.path.relpath(dup['path'], DATA_DIR_RAW)
        parts = rel_path.split(os.sep)
        if len(parts) >= 2:
            cls, filename = parts[0], parts[1]
            duplicate_set.add((cls, filename))

    files_to_remove = corrupt_set | duplicate_set

    # 3. Hapus file yang ditandai dari folder train dan val
    removed_count = 0
    for split_subfolder in ['train', 'val']:
        split_path = os.path.join(DATA_DIR_SPLIT, split_subfolder)
        if not os.path.isdir(split_path):
            continue
        for cls, filename in files_to_remove:
            target = os.path.join(split_path, cls, filename)
            if os.path.exists(target):
                os.remove(target)
                removed_count += 1

    print(f" Total file dihapus dari train/val : {removed_count}")

def normalize_extensions():

    jpeg_extensions = {'.jpeg', '.JPEG'}
    renamed_count = 0
    skipped = []

    for split_subfolder in ['train', 'val']:
        split_path = os.path.join(DATA_DIR_SPLIT, split_subfolder)
        if not os.path.isdir(split_path):
            continue
        for root, dirs, files in os.walk(split_path):
            for file in files:
                base, ext = os.path.splitext(file)
                if ext in jpeg_extensions:
                    old_path = os.path.join(root, file)
                    new_path = os.path.join(root, base + '.jpg')
                    # Jika file .jpg sudah ada, lewati
                    if os.path.exists(new_path):
                        skipped.append(old_path)
                        continue
                    os.rename(old_path, new_path)
                    renamed_count += 1
    
    print(f" file .jpeg diubah ke .jpg         : {renamed_count} ")

def balancing_data(dataset):
 
    classes = dataset.classes
    num_classes = len(classes)
    targets = np.array(dataset.targets)           
    class_counts = np.bincount(targets, minlength=num_classes).astype(float)
    total_samples = len(targets)
 
    # Bobot per kelas
    class_weights = total_samples / (num_classes * class_counts)
 
    # Bobot per sampel
    sample_weights = class_weights[targets] 
 
    # Sampler
    sampler = WeightedRandomSampler(
        weights = torch.FloatTensor(sample_weights),
        num_samples = len(sample_weights),
        replacement = True
    )
 
    # Tensor bobot untuk loss function 
    class_weights_tensor = torch.FloatTensor(class_weights)
 
    for i, cls_name in enumerate(classes):
        print(f"  -{cls_name:<55} : {int(class_counts[i]):>5} gambar  ->  bobot: {class_weights[i]:.4f}")
 
    return class_weights_tensor, sampler

def get_preprocess():

    # Augmentasi 

    train_transform = transforms.Compose([
        # 1. Resize ke 244px 
        transforms.Resize(224),
        # 2. Horizontal flip 
        transforms.RandomHorizontalFlip(p=0.5),
        # 3. ColorJitter: atasi perbedaan kecerahan dan kontras
        transforms.ColorJitter(
            brightness=0.2,  
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        # 4. Konversi ke tensor (skala [0,1])
        transforms.ToTensor(),
        # 5. Normalisasi statistik ImageNet (pretrained MobileNet)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Transform Validasi
    val_transform = transforms.Compose([
        # 1. Resize ke 244 px
        transforms.Resize(224),
        # 2. Konversi tensor
        transforms.ToTensor(),
        # 3. Normalisasi 
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform

def run_preprocess(eda_results):

    clean_dataset(
        df=None, 
        duplicates=eda_results.get('duplicates', []),
        corrupt_files=eda_results.get('corrupt_files', [])
    )

    normalize_extensions()

    train_transform, val_transform = get_preprocess()

    return train_transform, val_transform