import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from pathlib import Path
import os

def setup_output_dir():
    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def plot_scatter_pca(df, labels_map, output_dir):
    """
    Membuat scatter plot 2D dari hasil PCA.
    """
    plt.figure(figsize=(10, 6))
    
    # Map cluster ID to Sentiment Name
    df['Sentimen'] = df['Cluster'].map(labels_map)
    
    palette = {'Positif': 'green', 'Netral': 'gray', 'Negatif': 'red'}
    # Jika label gagal, fallback ke warna default seaborn
    if not all(s in palette for s in df['Sentimen'].unique()):
        palette = "viridis"

    sns.scatterplot(
        x='PCA1', y='PCA2',
        hue='Sentimen',
        palette=palette,
        data=df,
        alpha=0.6,
        s=50
    )
    plt.title('Scatter Plot 2D K-Means Clustering (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Sentimen')
    
    filepath = output_dir / 'cluster_scatter.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tersimpan: {filepath}")

def plot_sentiment_distribution(df, output_dir):
    """
    Membuat Pie Chart distribusi persentase sentimen.
    """
    counts = df['Sentimen'].value_counts()
    
    plt.figure(figsize=(8, 8))
    colors = ['#2ca02c', '#7f7f7f', '#d62728'] # Green, Gray, Red (approximate depending on order)
    # Match colors exactly if possible
    color_map = {'Positif': '#2ca02c', 'Netral': '#7f7f7f', 'Negatif': '#d62728'}
    pie_colors = [color_map.get(lbl, '#1f77b4') for lbl in counts.index]
    
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=pie_colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    plt.title('Distribusi Sentimen Ulasan Pelanggan')
    
    filepath = output_dir / 'distribusi_sentimen.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tersimpan: {filepath}")

def plot_wordclouds(df, cleaned_texts, labels_map, output_dir):
    """
    Membuat 3 Word Cloud untuk masing-masing sentimen.
    """
    df['Cleaned_Text'] = cleaned_texts
    
    for cluster_id, sentimen in labels_map.items():
        text_data = " ".join(df[df['Cluster'] == cluster_id]['Cleaned_Text'].tolist())
        
        if not text_data.strip():
            continue
            
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis' if sentimen == 'Netral' else ('Reds' if sentimen == 'Negatif' else 'Greens'),
            max_words=100
        ).generate(text_data)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - {sentimen}', fontsize=16)
        
        filename = f'wordcloud_{sentimen.lower()}.png'
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Tersimpan: {filepath}")
