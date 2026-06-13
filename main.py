import os
import sys

# Ensure root project path is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.loader import load_data
from modules.preprocessor import get_text_column, prepare_features
from modules.clustering import boost_sentiment_features, run_kmeans, get_top_keywords, auto_label_clusters, reassign_by_sentiment, get_representative_reviews, reduce_dimensions_pca
from modules.visualizer import setup_output_dir, plot_scatter_pca, plot_sentiment_distribution, plot_wordclouds

def main():
    print("="*75)
    print("   CUSTOMER REVIEW ANALYSIS — SENTIMENT CLUSTERING (MAJOR REVISION)")
    print("="*75)
    
    # 1. Inisialisasi folder output
    output_dir = setup_output_dir()
    
    # 2. Load Data
    filename = input("\nMasukkan nama file data CSV (di dalam folder big_data/): ").strip()
    if not filename.endswith('.csv'):
        filename += '.csv'
        
    df = load_data(filename)
    
    # 3. Sampling OOM Protection
    SAMPLE_SIZE = 10000
    if len(df) > SAMPLE_SIZE:
        print(f"\nData terlalu besar ({len(df):,} baris). Melakukan sampling acak {SAMPLE_SIZE:,} baris untuk mencegah Out of Memory (OOM)...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    
    # 4. Pilih Fitur
    text_col = get_text_column(df)
    if not text_col:
        print("Error: Tidak ada kolom teks yang dipilih. Program dihentikan.")
        return
        
    # 5. Preprocessing & Ekstraksi Fitur (Negation, Stopwords, TF-IDF)
    print("\n[Tahap 1] Memproses Teks...")
    X_tfidf, vectorizer, cleaned_texts = prepare_features(df, text_col)
    
    # 6. Clustering (K-Means K=2)
    print("\n[Tahap 2] Clustering...")
    k = 2
    
    # Boost fitur sentimen agar K-Means mengelompokkan berdasarkan EMOSI, bukan TOPIK
    print("Melakukan Sentiment Feature Boosting...")
    X_boosted = boost_sentiment_features(X_tfidf, vectorizer)
    
    labels, centroids, kmeans_model = run_kmeans(X_boosted, k)
    df['Cluster'] = labels
    
    # Reduksi Dimensi untuk Visualisasi (PCA 2D)
    pca1, pca2 = reduce_dimensions_pca(X_tfidf)
    df['PCA1'] = pca1
    df['PCA2'] = pca2
    
    # 7. Pelabelan Sentimen Otomatis & Ekstraksi Insight
    print("\n[Tahap 3] Ekstraksi Insight & Pelabelan Sentimen...")
    cluster_keywords = get_top_keywords(X_tfidf, labels, vectorizer, k)
    labels_map = auto_label_clusters(cluster_keywords)
    
    # Post-Clustering Reassignment: pindahkan review yang jelas salah cluster
    print("\nMelakukan Post-Clustering Reassignment...")
    labels = reassign_by_sentiment(labels, df, text_col, labels_map)
    df['Cluster'] = labels
    
    # Hitung ulang keywords & representative reviews dengan label yang sudah diperbaiki
    cluster_keywords = get_top_keywords(X_tfidf, labels, vectorizer, k)
    
    rep_reviews = get_representative_reviews(df, text_col, X_tfidf, labels, kmeans_model, labels_map, k)

    
    # Cetak Laporan Teks
    report_text = ["=== LAPORAN CLUSTERING SENTIMEN ===\n"]
    for cluster_id in range(k):
        sentimen = labels_map.get(cluster_id, f"Cluster {cluster_id}")
        count = (df['Cluster'] == cluster_id).sum()
        
        report_text.append(f"[{sentimen.upper()}] - {count} Ulasan")
        report_text.append(f"Top Keywords: {', '.join(cluster_keywords[cluster_id])}")
        report_text.append("Representative Reviews:")
        for idx, review in enumerate(rep_reviews.get(cluster_id, [])):
            report_text.append(f"  {idx+1}. {review}")
        report_text.append("-" * 50)
        
    report_content = "\n".join(report_text)
    print("\n" + report_content)
    
    # Simpan Laporan Teks
    report_filepath = output_dir / 'laporan_insight_sentimen.txt'
    with open(report_filepath, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"\nLaporan teks disimpan di: {report_filepath}")
        
    # 8. Visualisasi
    print("\n[Tahap 4] Membuat Visualisasi...")
    plot_scatter_pca(df, labels_map, output_dir)
    plot_sentiment_distribution(df, output_dir)
    plot_wordclouds(df, cleaned_texts, labels_map, output_dir)
    
    print("\nSelesai! Seluruh grafik dan laporan telah disimpan di folder 'output/'.")
    print("="*75)

if __name__ == "__main__":
    main()
