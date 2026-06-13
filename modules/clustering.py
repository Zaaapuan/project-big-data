import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

# ============================================================
# SENTIMENT LEXICON (Digunakan untuk Boosting & Auto-Labeling)
# ============================================================
POSITIVE_WORDS = {
    # Indonesia
    'bagus', 'baik', 'puas', 'cepat', 'ramah', 'mantap', 'keren', 'suka',
    'rekomen', 'aman', 'terbaik', 'top', 'sempurna', 'memuaskan', 'senang',
    'nyaman', 'mulus', 'berkualitas', 'hebat', 'sip',
    # English
    'good', 'great', 'awesome', 'excellent', 'best', 'perfect', 'satisfied',
    'amazing', 'love', 'recommend', 'nice', 'fantastic', 'wonderful', 'happy',
    'smooth', 'quality', 'superb', 'impressive', 'outstanding', 'positive',
    'pleasant', 'enjoy', 'fast', 'reliable', 'ok', 'okay', 'thanks', 'thank', 'wow', 'worth', 'recommended',
}

NEGATIVE_WORDS = {
    # Indonesia
    'buruk', 'jelek', 'kecewa', 'lambat', 'rusak', 'parah', 'hancur',
    'mahal', 'bohong', 'palsu', 'cacat', 'cacad', 'menipu', 'sampah',
    # English
    'bad', 'terrible', 'worst', 'awful', 'slow', 'disappointed', 'broken',
    'hate', 'poor', 'horrible', 'defective', 'fraud', 'fake', 'useless',
    'rubbish', 'cheap', 'inferior', 'fail', 'waste', 'damage', 'disgust',
    'negative', 'ugly', 'annoying', 'worthless', 'not_good', 'not_recommended',
}

# Boost factor: seberapa kuat kata sentimen diboost di matrix TF-IDF
# Semakin besar nilainya, K-Means semakin "terpaksa" memisah berdasarkan sentimen
SENTIMENT_BOOST_FACTOR = 10.0


def boost_sentiment_features(X_tfidf, vectorizer):
    """
    Teknik utama agar K-Means bekerja berdasarkan SENTIMEN, bukan TOPIK.
    
    Cara kerja:
    - Setelah TF-IDF dibuat, kalikan kolom-kolom yang berisi kata sentimen
      (positif / negatif) dengan faktor penguat (SENTIMENT_BOOST_FACTOR).
    - Dengan bobot yang sangat besar pada kata sentimen, K-Means menjadi
      sangat sensitif terhadap kata emosi dan memisahkan cluster berdasarkan
      sentimen, bukan topik.
    """
    feature_names = vectorizer.get_feature_names_out()
    X_boosted = X_tfidf.astype(float).copy()

    boosted_count = 0
    for idx, word in enumerate(feature_names):
        # Cek apakah kata ini ada di lexicon sentimen
        word_base = word.split('_')[0]  # Handle kata negasi seperti "tidak_bagus"
        is_positive = (word in POSITIVE_WORDS or word_base in POSITIVE_WORDS)
        is_negative = (word in NEGATIVE_WORDS or word_base in NEGATIVE_WORDS)

        if is_positive or is_negative:
            X_boosted[:, idx] *= SENTIMENT_BOOST_FACTOR
            boosted_count += 1

    print(f"  -> {boosted_count} fitur sentimen diboost dengan faktor {SENTIMENT_BOOST_FACTOR}x")
    return X_boosted


def run_kmeans(X_boosted, k=2):
    """
    Run K-Means clustering dengan K=2 pada matrix yang sudah di-boost.
    """
    print(f"Menjalankan K-Means Clustering dengan K={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_boosted)
    return labels, kmeans.cluster_centers_, kmeans


def reassign_by_sentiment(labels, df, text_col, labels_map):
    """
    Post-Clustering Reassignment:
    Setelah K-Means selesai, setiap review di-skor secara individual.
    Jika skor sentimennya JELAS BERLAWANAN dengan label clusternya,
    review tersebut dipindahkan ke cluster yang labelnya cocok.
    
    Contoh:
      - Review "great, perfect, satisfied" masuk cluster NEGATIF → DIPINDAH ke cluster POSITIF
      - Review "jelek, rusak" masuk cluster POSITIF → DIPINDAH ke cluster NEGATIF
    
    Review dengan sentimen NETRAL (skor seimbang) tidak dipindah.
    """
    # Buat reverse map hanya untuk label Positif & Negatif
    reverse_map = {v: k for k, v in labels_map.items() if v in ('Positif', 'Negatif')}
    
    new_labels = labels.copy()
    reassigned = 0
    
    for i, row_text in enumerate(df[text_col]):
        current_cluster = labels[i]
        current_sentiment = labels_map.get(current_cluster, '')
        
        # Hitung skor sentimen individual
        words = str(row_text).lower().split()
        pos_score = sum(1 for w in words if w in POSITIVE_WORDS)
        neg_score = sum(1 for w in words if w in NEGATIVE_WORDS)
        
        # Hanya reassign jika ada sinyal sentimen yang jelas (selisih >= 1)
        if pos_score - neg_score >= 1:
            individual_sentiment = 'Positif'
        elif neg_score - pos_score >= 1:
            individual_sentiment = 'Negatif'
        else:
            # Sentimen tidak jelas → jangan dipindah
            continue
        
        # Pindahkan jika label cluster tidak cocok dengan sentimen individu
        if current_sentiment != individual_sentiment:
            target_cluster = reverse_map.get(individual_sentiment)
            if target_cluster is not None:
                new_labels[i] = target_cluster
                reassigned += 1
    
    print(f"  -> Post-Clustering Reassignment: {reassigned} ulasan dipindah ke cluster yang lebih sesuai")
    return new_labels



def get_top_keywords(X_tfidf, labels, vectorizer, k=3, top_n=10):
    """
    Mendapatkan top keywords untuk setiap cluster.
    Menggunakan X_tfidf ASLI (bukan yang di-boost) agar keyword yang ditampilkan tetap natural.
    """
    df_tfidf = pd.DataFrame(X_tfidf.todense(), columns=vectorizer.get_feature_names_out())
    df_tfidf['Cluster'] = labels

    cluster_keywords = {}
    for i in range(k):
        cluster_data = df_tfidf[df_tfidf['Cluster'] == i].drop('Cluster', axis=1)
        mean_tfidf = cluster_data.mean(axis=0)
        top_words = mean_tfidf.nlargest(top_n).index.tolist()
        cluster_keywords[i] = top_words

    return cluster_keywords


def auto_label_clusters(cluster_keywords):
    """
    Melabeli cluster menjadi Positif atau Negatif (binary).
    
    Logika scoring:
    - Setiap kata sentimen positif  → skor += 1
    - Setiap kata sentimen negatif  → skor -= 1
    - n‑gram yang mengandung kata sentimen → skor += 0.5 / -0.5
    
    Logika labeling:
    - Urutkan cluster dari skor tertinggi ke terendah
    - Cluster tertinggi  → Positif
    - Cluster terendah   → Negatif
    - Jika hanya satu cluster (fallback) → beri label Positif
    """
    scores = {}
    for cluster_id, words in cluster_keywords.items():
        pos_score = 0.0
        neg_score = 0.0
        for w in words:
            word_base = w.split('_')[0]  # handle negasi: tidak_bagus → tidak
            if w in POSITIVE_WORDS or word_base in POSITIVE_WORDS:
                pos_score += 1
            elif w in NEGATIVE_WORDS or word_base in NEGATIVE_WORDS:
                neg_score += 1
            elif any(p in w for p in POSITIVE_WORDS):
                pos_score += 0.5
            elif any(n in w for n in NEGATIVE_WORDS):
                neg_score += 0.5
        diff = pos_score - neg_score
        scores[cluster_id] = {'pos': pos_score, 'neg': neg_score, 'diff': diff}
        print(f"  Cluster {cluster_id}: pos={pos_score:.1f}, neg={neg_score:.1f}, diff={diff:+.1f} | keywords: {words[:5]}")

    # Urutkan: tertinggi → Positif, terendah → Negatif
    sorted_clusters = sorted(scores.items(), key=lambda x: x[1]['diff'], reverse=True)

    labels_map = {}
    if len(sorted_clusters) >= 2:
        labels_map[sorted_clusters[0][0]] = 'Positif'
        labels_map[sorted_clusters[-1][0]] = 'Negatif'
        # Jika ada cluster ketiga (mis‑config), beri label Netral sementara
        if len(sorted_clusters) == 3:
            labels_map[sorted_clusters[1][0]] = 'Netral'
    elif len(sorted_clusters) == 1:
        labels_map[sorted_clusters[0][0]] = 'Positif'
    else:
        # fallback: assign generic names
        for cluster_id in cluster_keywords.keys():
            labels_map[cluster_id] = f"Cluster {cluster_id}"

    return labels_map


def score_single_review(text):
    """
    Menghitung skor sentimen (+/-) untuk satu teks ulasan secara individual.
    Digunakan untuk memvalidasi apakah ulasan cocok dengan label cluster-nya.
    Returns: 'Positif', 'Negatif'
    """
    words = str(text).lower().split()
    pos_score = 0
    neg_score = 0
    for word in words:
        if word in POSITIVE_WORDS:
            pos_score += 1
        elif word in NEGATIVE_WORDS:
            neg_score += 1
def get_representative_reviews(df, text_col, X_tfidf, labels, kmeans, labels_map, k=2, top_n=3):
    """
    Mendapatkan review paling dekat dengan centroid untuk divalidasi.
    Setiap kandidat divalidasi skor sentimennya agar cocok dengan label cluster.
    Jika tidak ada yang cocok, fallback ke kandidat terdekat tanpa filter.
    """
    rep_reviews = {}
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue

        cluster_points = X_tfidf[cluster_indices]
        centroid = np.asarray(cluster_points.mean(axis=0))
        distances = euclidean_distances(cluster_points, centroid).flatten()

        n_candidates = min(len(cluster_indices), top_n * 3)
        candidate_idx = distances.argsort()[:n_candidates]
        candidate_original_idx = cluster_indices[candidate_idx]

        expected_sentiment = labels_map.get(i, '')

        # Filter: hanya tampilkan review yang skor sentimennya sesuai label cluster
        validated = []
        fallback = []
        for orig_idx in candidate_original_idx:
            review_text = df.iloc[orig_idx][text_col]
            review_sentiment = score_single_review(review_text)
            is_match = (review_sentiment == expected_sentiment)
            if is_match:
                validated.append(review_text)
            else:
                fallback.append(review_text)
            if len(validated) >= top_n:
                break
        # Jika tidak ada yang cocok (dataset sangat kecil), gunakan fallback
        if not validated:
            validated = fallback[:top_n]
        rep_reviews[i] = validated[:top_n]

    return rep_reviews


def score_single_review(text):
    """
    Menghitung skor sentimen (+/-) untuk satu teks ulasan secara individual.
    Returns: 'Positif' atau 'Negatif'.
    """
    words = str(text).lower().split()
    pos_score = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_score = sum(1 for w in words if w in NEGATIVE_WORDS)
    if pos_score > neg_score:
        return 'Positif'
    else:
        return 'Negatif'


def reduce_dimensions_pca(X_tfidf):
    """
    Mengubah fitur TF-IDF kompleks menjadi 2D menggunakan PCA untuk scatter plot.
    """
    print("Mereduksi dimensi data menjadi 2D dengan PCA untuk visualisasi...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_tfidf.toarray())
    return X_pca[:, 0], X_pca[:, 1]
