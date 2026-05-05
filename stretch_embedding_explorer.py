"""
Module 6 Week B — Stretch: Embedding Space Explorer
Visualization of GloVe and DistilBERT embeddings using t-SNE.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from embeddings_lab import load_glove, text_to_glove, extract_bert_embedding
from transformers import AutoTokenizer, AutoModel

# 1. Setup Data for Word Visualization (GloVe)
# We define 5 categories and some words for each
word_categories = {
    "Countries": ["london", "paris", "tokyo", "beijing", "cairo", "rome", "madrid", "amman", "berlin", "ottawa"],
    "Sports": ["football", "basketball", "tennis", "soccer", "stadium", "athlete", "referee", "olympics", "goal", "coach"],
    "Tech": ["computer", "software", "iphone", "internet", "robot", "algorithm", "hacker", "database", "digital", "battery"],
    "Finance": ["bank", "money", "economy", "market", "stock", "investment", "dollar", "finance", "profit", "budget"],
    "Emotions": ["happy", "sad", "angry", "joy", "fear", "love", "excited", "anxious", "brave", "lonely"]
}

def plot_embeddings(embeddings, labels, categories, title, filename):
    """Creates a 2D scatter plot of embeddings."""
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='pca', learning_rate='auto')
    reduced_data = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    
    # Create color map for categories
    unique_cats = list(set(categories))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_cats)))
    
    for i, cat in enumerate(unique_cats):
        indices = [j for j, c in enumerate(categories) if c == cat]
        plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1], 
                    label=cat, color=colors[i], alpha=0.7, edgecolors='k')

    # Add annotations for a few points
    for i, label in enumerate(labels):
        if i % 3 == 0:  # Annotate every 3rd word to keep it clean
            plt.annotate(label, (reduced_data[i, 0], reduced_data[i, 1]), 
                         fontsize=9, alpha=0.8, xytext=(5, 2), textcoords='offset points')

    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

if __name__ == "__main__":
    # --- Part A: GloVe Word Clusters ---
    print("Processing GloVe word embeddings...")
    glove = load_glove("data/glove_50k_50d.txt")
    
    all_words = []
    all_word_vecs = []
    word_labels = []
    
    for cat, words in word_categories.items():
        for w in words:
            if w in glove:
                all_words.append(w)
                all_word_vecs.append(glove[w])
                word_labels.append(cat)
    
    plot_embeddings(np.array(all_word_vecs), all_words, word_labels, 
                    "GloVe Word Embedding Clusters (t-SNE)", "word_clusters.png")

    # --- Part B: BBC News Document Clusters ---
    print("\nProcessing DistilBERT document embeddings...")
    df = pd.read_csv("data/bbc_news.csv")
    # Sample 20 articles (4 from each category)
    sample_df = df.groupby('category').head(4).reset_index()
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    
    doc_vecs = [extract_bert_embedding(text, tokenizer, model) for text in sample_df['text']]
    doc_labels = sample_df['category'].tolist()
    doc_titles = [t[:20] for t in sample_df['text']] # Short version of text for labels

    plot_embeddings(np.array(doc_vecs), doc_titles, doc_labels, 
                    "BBC News DistilBERT Clusters (t-SNE)", "document_clusters.png")