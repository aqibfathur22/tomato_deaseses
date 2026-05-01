from src.split_data import run_split
from src.eda import run_eda
from src.data_loader import get_loaders

def main():
    # 1. Split Data
    run_split()

    # 2. EDA
    print("\n!!! Running : Exploratory Data Analysis !!!")
    run_eda()

    # 3. Preprocessing dan load data
    print("\n!!! Running : Data Loaders !!!")
    train_loader, val_loader, train_ds, val_ds, classes = get_loaders()

    print(f"Class: {classes}\n")
    print(f"Batch Training: {len(train_loader)}")
    print(f"Batch Validasi: {len(val_loader)}\n")
    print(f"Train: {len(train_ds)}")
    print(f"Validasi: {len(val_ds)}")

if __name__ == "__main__":
    main()

