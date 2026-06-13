import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import sys
import os

# Konfigurasi
COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
FIGURE_SIZE = (10, 6)
SAMPLE_SIZE = 10000
DATA_DIR = Path(__file__).parent / "big_data"
VISUALIZATION_DIR = Path(__file__).parent / "Hasil Visualisasi"

# Membuat folder Hasil Visualisasi jika belum ada
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

def load_data(filename):
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"Error: File '{filepath}' tidak ditemukan.")
        print(f"Pastikan file ada di folder {DATA_DIR}/")
        sys.exit(1)
    # Menambahkan thousands=',' dan na_values=['-'] agar angka ribuan dan nilai kosong ('-') terbaca benar
    df = pd.read_csv(filepath, thousands=',', na_values=['-'])
    
    # Hapus baris yang mengandung NaN agar tidak error saat clustering
    if df.isna().any().any():
        print(f"\nDitemukan missing values (NaN), menghapus baris yang kosong...")
        df = df.dropna().reset_index(drop=True)
        
    return df

def preview_data(df):
    print("\n" + "="*60)
    print("PREVIEW DATA (5 baris pertama):")
    print("="*60)
    print(df.head().to_string())
    print("\n" + "="*60)
    print("INFO DATASET:")
    print("="*60)
    print(f"Jumlah baris: {len(df):,}")
    print(f"Jumlah kolom: {len(df.columns)}")
    print(f"\nKolom dan tipe data:")
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        print(f"  {col}: {dtype} (non-null: {non_null:,})")
    print()

def get_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def sample_data(df):
    if len(df) > SAMPLE_SIZE:
        print(f"Data terlalu besar ({len(df):,} baris). Sampling {SAMPLE_SIZE:,} baris untuk visualisasi...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
    return df

def get_user_input(df):
    numeric_cols = get_numeric_columns(df)
    print("Kolom numerik yang tersedia:")
    for i, col in enumerate(numeric_cols, 1):
        print(f"  {i}. {col}")

    while True:
        input_indices = input("\nMasukkan nomor urut kolom-kolom untuk clustering (pisahkan dengan koma, minimal 2): ").strip()
        selected_indices = [idx.strip() for idx in input_indices.split(',')]
        
        if len(selected_indices) < 2:
            print("Error: Harap masukkan minimal 2 kolom untuk clustering.")
            continue
            
        selected_cols = []
        valid = True
        for idx_str in selected_indices:
            try:
                idx = int(idx_str)
                if 1 <= idx <= len(numeric_cols):
                    selected_cols.append(numeric_cols[idx - 1])
                else:
                    print(f"Error: Nomor kolom '{idx}' di luar rentang (1 - {len(numeric_cols)}).")
                    valid = False
                    break
            except ValueError:
                print(f"Error: '{idx_str}' bukan nomor kolom yang valid.")
                valid = False
                break
                
        if valid:
            break

    while True:
        try:
            k = int(input("Masukkan jumlah cluster (K): ").strip())
            if k < 2:
                print("Error: Jumlah cluster minimal 2.")
            elif k > len(COLORS):
                print(f"Error: Maksimal {len(COLORS)} cluster.")
            else:
                break
        except ValueError:
            print("Error: Masukkan angka yang valid.")

    return selected_cols, k

def run_clustering(df, selected_cols, k):
    print(f"\nMenyiapkan data ({len(selected_cols)} kolom) dan melakukan Scaling...")
    X_raw = df[selected_cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    print(f"Menjalankan MiniBatchKMeans dengan K={k}...")
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=256,
        random_state=42,
        n_init='auto'
    )
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    print("Mereduksi dimensi data menggunakan PCA ke dalam 2 Komponen Utama (PC1 & PC2)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]
    
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    
    col_info = ", ".join(selected_cols)
    if len(selected_cols) > 3:
        col_info = f"{selected_cols[0]}, {selected_cols[1]}, ... ({len(selected_cols)} fitur)"
        
    return df, centroids_pca, col_info

def show_cluster_counts(df):
    print("\n" + "="*60)
    print("JUMLAH ANGGOTA SETIAP CLUSTER:")
    print("="*60)
    counts = df['Cluster'].value_counts().sort_index()
    for cluster_id, count in counts.items():
        print(f"  Cluster {cluster_id}: {count:,} anggota")
    print()

def plot_clusters(df, centroids_pca, k, col_info):
    plt.figure(figsize=FIGURE_SIZE)
    sns.set_style("whitegrid")

    for i in range(k):
        cluster_data = df[df['Cluster'] == i]
        plt.scatter(
            cluster_data['PC1'],
            cluster_data['PC2'],
            c=COLORS[i],
            label=f'Cluster {i}',
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
            s=50
        )

    plt.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        c='red',
        marker='X',
        s=200,
        linewidths=3,
        edgecolors='black',
        label='Centroid',
        zorder=5
    )

    plt.xlabel("Principal Component 1 (PC1)", fontsize=12)
    plt.ylabel("Principal Component 2 (PC2)", fontsize=12)
    plt.title(f'K-Means Clustering (K={k}) menggunakan PCA 2D\nFitur: {col_info}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()

    # Logika penamaan dinamis untuk mencegah overwrite
    base_name = "hasil_clustering"
    ext = ".png"
    filepath = VISUALIZATION_DIR / f"{base_name}{ext}"
    counter = 1
    while filepath.exists():
        filepath = VISUALIZATION_DIR / f"{base_name} ({counter}){ext}"
        counter += 1

    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\nVisualisasi tersimpan sebagai: {filepath}")

    # Buka otomatis (Windows)
    if os.name == 'nt':
        os.startfile(filepath)

    plt.show()

def main():
    print("="*60)
    print("   K-MEANS CLUSTERING UNTUK BIG DATA (PCA MULTI-KOLOM)")
    print("="*60)

    filename = input("\nMasukkan nama file CSV (di folder big_data/): ").strip()
    df = load_data(filename)
    preview_data(df)
    df = sample_data(df)
    selected_cols, k = get_user_input(df)
    df_labeled, centroids_pca, col_info = run_clustering(df, selected_cols, k)
    show_cluster_counts(df_labeled)
    plot_clusters(df_labeled, centroids_pca, k, col_info)
    print("\nSelesai! Hasil clustering telah disimpan.")

if __name__ == "__main__":
    main()
