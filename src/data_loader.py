from torchvision import datasets
from torch.utils.data import DataLoader
import os

from .config import BATCH_SIZE, DATA_DIR_SPLIT
from .preprocess import get_preprocess, balancing_data

def get_loaders():
    
    # get preprocess
    train_transform, val_transform = get_preprocess()

    # path data
    train_path = os.path.join(DATA_DIR_SPLIT, 'train')
    val_path = os.path.join(DATA_DIR_SPLIT, 'val')

    # load data
    train_ds = datasets.ImageFolder(train_path, transform=train_transform)
    val_ds = datasets.ImageFolder(val_path, transform=val_transform)
    
    # Sampler dan class_weights
    class_weights, sampler = balancing_data(train_ds)


    train_loader = DataLoader(
        train_ds,
        batch_size  = BATCH_SIZE,
        sampler     = sampler,   
        num_workers = 2,
        pin_memory  = True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True
    )
    
    return train_loader, val_loader, train_ds, val_ds, train_ds.classes, class_weights