import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

import nltk
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Combine English (NLTK) and Indonesian (Sastrawi) stopwords
try:
    nltk.data.find('corpora/stopwords')
    stop_words_eng = set(stopwords.words('english'))
except LookupError:
    print(
        "Peringatan: Corpus stopwords NLTK tidak tersedia. "
        "Menggunakan stopwords bahasa Inggris bawaan scikit-learn."
    )
    stop_words_eng = set(ENGLISH_STOP_WORDS)
factory = StopWordRemoverFactory()
stop_words_ind = set(factory.get_stop_words())

# Custom domain-specific stopwords: kata topik/umum yang bukan sentimen
# dan sering muncul sehingga mengacaukan clustering sentimen
CUSTOM_DOMAIN_STOPWORDS = {
    # Kata umum Indonesia yang bukan sentimen
    'barang', 'barangnya', 'produk', 'produknya', 'item',
    'harga', 'harganya', 'pengiriman', 'ongkir',
    'sangat', 'sekali', 'lumayan', 'cukup', 'agak',
    'untuk', 'dengan', 'saat', 'sudah', 'telah', 'masih',
    'ini', 'itu', 'saja', 'juga', 'bisa', 'ada', 'tidak_ada',
    'tapi', 'tapi', 'namun', 'memang',
    # Kata umum Inggris yang bukan sentimen
    'product', 'item', 'purchase', 'bought', 'buy',
    'price', 'shipping', 'delivery', 'arrived', 'arrival',
    'customer', 'service', 'build', 'quality', 'performance',
    'use', 'used', 'using', 'time', 'day', 'month',
    'make', 'made', 'get', 'got',
}

combined_stopwords = list(stop_words_eng.union(stop_words_ind).union(CUSTOM_DOMAIN_STOPWORDS))

def get_text_column(df):
    """
    Meminta user memilih kolom teks review utama.
    """
    text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if not text_cols:
        print("Error: Tidak ditemukan kolom bertipe teks (object/string) dalam data.")
        return None
        
    print("\nKolom teks yang tersedia:")
    for i, col in enumerate(text_cols, 1):
        print(f"  {i}. {col}")
        
    while True:
        try:
            choice = input("\nMasukkan nomor urut kolom teks utama (review): ").strip()
            idx = int(choice)
            if 1 <= idx <= len(text_cols):
                selected = text_cols[idx - 1]
                print(f"Kolom teks yang dipilih: '{selected}'")
                return selected
            else:
                print(f"Error: Nomor harus berada dalam rentang 1 - {len(text_cols)}.")
        except ValueError:
            print("Error: Harap masukkan angka yang valid.")

def clean_text(text):
    """
    Membersihkan satu string teks mentah.
    Termasuk penanganan negasi (negation handling).
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Penanganan Negasi (Negation Handling)
    # Menggabungkan kata "tidak", "bukan", "kurang" dengan kata setelahnya
    text = re.sub(r'\b(tidak|bukan|kurang|jangan)\s+(\w+)\b', r'\1_\2', text)
    
    # 3. Hapus angka
    text = re.sub(r'\d+', '', text)
    
    # 4. Hapus tanda baca dan karakter khusus (mempertahankan underscore untuk negasi)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 5. Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def prepare_features(df, text_col):
    """
    Mengubah teks menjadi matrix TF-IDF dan menghapus stopwords.
    """
    print("Melakukan pembersihan teks (Cleaning & Negation Handling)...")
    cleaned_texts = [clean_text(t) for t in df[text_col]]
    
    print("Melakukan ekstraksi TF-IDF dan Stopwords Removal...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=combined_stopwords,
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    
    # Return cleaned texts as well to update dataframe if needed
    return tfidf_matrix, vectorizer, cleaned_texts
