import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import hashlib
from .config import DATA_DIR_RAW


def run_eda():
    classes = sorted([
        c for c in os.listdir(DATA_DIR_RAW)
        if os.path.isdir(os.path.join(DATA_DIR_RAW, c))
    ])
    data = []
    hashes = {}
    duplicates = []
    corrupt_files = []

    # PENGUMPULAN METADATA SETIAP FILE

    for cls in classes:
        cls_path = os.path.join(DATA_DIR_RAW, cls)
        for f in sorted(os.listdir(cls_path)):
            f_path = os.path.join(cls_path, f)
            ext = f.rsplit('.', 1)[-1].lower() if '.' in f else 'unknown'

            try:
                # Hash untuk deteksi duplikat
                file_hash = hashlib.md5(open(f_path, 'rb').read()).hexdigest()
                is_duplicate = file_hash in hashes
                if is_duplicate:
                    duplicates.append({'path': f_path, 'original': hashes[file_hash]})
                else:
                    hashes[file_hash] = f_path

                with Image.open(f_path) as img:
                    width, height = img.size
                    mode = img.mode

                # Kecerahan rata-rata via grayscale
                img_cv = cv2.imread(f_path)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                brightness = float(np.mean(gray))

                data.append({
                    'class': cls,
                    'filename': f,
                    'path': f_path,
                    'width': width,
                    'height': height,
                    'resolution': f"{width}x{height}",
                    'format': ext,
                    'mode': mode,
                    'brightness': brightness,
                    'hash':  file_hash,
                    'is_duplicate': is_duplicate,
                })

            except Exception as e:
                corrupt_files.append({'path': f_path, 'error': str(e)})

    df = pd.DataFrame(data)
    df_valid = df[~df['is_duplicate']].copy()

    # 1. DISTRIBUSI & CEK BALANCE ANTAR CLASS

    print("\n 1. DISTRIBUSI & BALANCE DATASET\n")

    class_counts = df_valid['class'].value_counts().sort_index()
    for cls, cnt in class_counts.items():
        print(f"  {cls:<50} : {cnt} gambar")

    min_count = class_counts.min()
    max_count = class_counts.max()
    imbalance_ratio = max_count / min_count
    print(f"\n  Imbalance Ratio (max/min): {imbalance_ratio:.2f}")

    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_counts.index, 
                y=class_counts.values, 
                hue=class_counts.index, 
                palette='viridis', 
                legend=False)
    plt.title("Distribusi Jumlah Gambar per Class")
    plt.ylabel("Jumlah Gambar")
    plt.xlabel("Class")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("eda_1_class_distribution.png", dpi=150)
    plt.show()

    # 2. VARIASI RESOLUSI (PIXEL)

    print("\n 2. VARIASI RESOLUSI GAMBAR\n")

    resolution_counts = df_valid['resolution'].value_counts()
    print(f"  Total variasi resolusi : {len(resolution_counts)}")
    for res, cnt in resolution_counts.head(10).items():
        print(f"  Resolusi Gambar        : {res} px  ({cnt} gambar)")

    # 3. DETEKSI FILE DUPLIKAT

    print("\n 3. DETEKSI FILE DUPLIKAT\n")

    print(f"  Gambar Duplikat : {len(duplicates)}")

    # 4. VARIASI FORMAT FILE

    print("\n 4. VARIASI FORMAT FILE\n")

    format_counts = df_valid['format'].value_counts()
    for fmt, cnt in format_counts.items():
        print(f"  .{fmt:<10} : {cnt} file")

    # 7. VARIASI KECERAHAN (BRIGHTNESS)

    print("\n 7. ANALISIS KECERAHAN GAMBAR\n")

    mean_brightness = df_valid['brightness'].mean()
    min_brightness = df_valid['brightness'].min()
    max_brightness = df_valid['brightness'].max()

    print(f"  Rata-rata Kecerahan : {mean_brightness:.2f}  (0=gelap, 255=terang)")
    print(f"  Min / Max           : {min_brightness:.2f} / {max_brightness:.2f}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(df_valid['brightness'], bins=50, color='gold', edgecolor='white')
    plt.title("Distribusi Kecerahan Gambar")
    plt.xlabel("Rata-rata Pixel (Grayscale)")
    plt.ylabel("Jumlah")

    plt.subplot(1, 2, 2)
    brightness_by_class = df_valid.groupby('class')['brightness'].mean().sort_values()
    sns.barplot(x=brightness_by_class.values, 
                y=brightness_by_class.index,
                palette='YlOrBr', 
                hue=class_counts.index,
                orient='h',
                legend=False)
    plt.title("Rata-rata Kecerahan per Class")
    plt.xlabel("Rata-rata Kecerahan")
    plt.tight_layout()
    plt.savefig("eda_7_brightness_distribution.png", dpi=150)
    plt.show()

    # RINGKASAN AKHIR
   
    print(" RINGKASAN HASIL EDA\n")

    print(f"  Total gambar valid         : {len(df_valid)}")
    print(f"  File rusak / tidak terbaca : {len(corrupt_files)}")
    print(f"  Duplikat terdeteksi        : {len(duplicates)}")
    print(f"  Variasi resolusi unik      : {len(resolution_counts)}")
    print(f"  Variasi format file        : {list(format_counts.index)}")
    print(f"  Variasi mode warna         : {list(df_valid['mode'].unique())}")
    print(f"  Imbalance ratio            : {imbalance_ratio:.2f}")
    print("=" * 60)

    return df_valid, {
        'class_counts': class_counts,
        'resolution_counts': resolution_counts,
        'format_counts': format_counts,
        'duplicates': duplicates,
        'corrupt_files':  corrupt_files,
        'imbalance_ratio': imbalance_ratio,
        'mean_brightness': mean_brightness,
    }


if __name__ == "__main__":
    run_eda()