import splitfolders
from .config import DATA_DIR_RAW, DATA_DIR_SPLIT

def run_split():
    # Train (80%), Val (10%), Test (10%)
    splitfolders.ratio(
        DATA_DIR_RAW, 
        output=DATA_DIR_SPLIT, 
        seed=42, 
        ratio=(.8, .1, .1), 
        group_prefix=None
    )
    print(f" split sukses : {DATA_DIR_SPLIT}")

if __name__ == "__main__":
    run_split()