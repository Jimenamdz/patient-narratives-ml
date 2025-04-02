import os
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    calinski_harabasz_score, davies_bouldin_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

# === Logging setup ===
logging.basicConfig(filename='clustering_analysis.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')
logging.info("Clustering analysis started.")

# Reproducibility
seed = 42
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(seed)

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Device
DEVICE = torch.device("cpu")

# Constants
BATCH_SIZE = 32
MAX_LENGTH = 128
FILE_PATH = "cleaned_patient_data.csv"

# Initialize tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEVICE)

# Compute embeddings
def get_sentence_embeddings(texts, batch_size=BATCH_SIZE):
    embeddings = []
    model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding sentences"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                           padding=True, max_length=MAX_LENGTH).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        batch_embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)

    return np.array(embeddings)

# Load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['group', 'text', 'source']
    terminal_texts = df[df['group'].str.lower() == 'terminal']['text'].dropna().tolist()
    non_terminal_texts = df[df['group'].str.lower() == 'non-terminal']['text'].dropna().tolist()
    return terminal_texts, non_terminal_texts

# Apply KMeans
def apply_kmeans(embeddings, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

if __name__ == "__main__":
    terminal_texts, non_terminal_texts = load_data(FILE_PATH)
    terminal_embeddings = get_sentence_embeddings(terminal_texts)
    non_terminal_embeddings = get_sentence_embeddings(non_terminal_texts)

    all_embeddings = np.vstack((terminal_embeddings, non_terminal_embeddings))
    labels_true = np.array(['Terminal'] * len(terminal_embeddings) + ['Non-Terminal'] * len(non_terminal_embeddings))

    # Hyperparameter optimization
    silhouette_scores = []
    k_values = range(2, 8)
    for k in k_values:
        clusters_temp = KMeans(n_clusters=k, random_state=seed).fit_predict(all_embeddings)
        silhouette_scores.append(silhouette_score(all_embeddings, clusters_temp))

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, silhouette_scores, 'o-')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Optimal Number of Clusters")
    plt.grid(True)
    plt.show()

    # Main clustering
    cluster_labels = apply_kmeans(all_embeddings, n_clusters=2)

    # Robustness check
    agg_labels = AgglomerativeClustering(n_clusters=2).fit_predict(all_embeddings)

    # Comprehensive evaluation
    silhouette_avg = silhouette_score(all_embeddings, cluster_labels)
    ari_score = adjusted_rand_score(labels_true, cluster_labels)
    ch_score = calinski_harabasz_score(all_embeddings, cluster_labels)
    db_score = davies_bouldin_score(all_embeddings, cluster_labels)

    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
    print(f"Calinski-Harabasz Index: {ch_score:.2f}")
    print(f"Davies-Bouldin Index: {db_score:.2f}")

    logging.info(f"Scores - Silhouette: {silhouette_avg}, ARI: {ari_score}, CH: {ch_score}, DB: {db_score}")

    # Cluster profile (TF-IDF)
    df_clusters = pd.DataFrame({
        'text': terminal_texts + non_terminal_texts,
        'cluster': cluster_labels,
        'sentiment': np.load('terminal_sentiment.npy').tolist() + np.load('non_terminal_sentiment.npy').tolist(),
        'true_label': labels_true
    })

    vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
    for cluster in np.unique(cluster_labels):
        texts_in_cluster = df_clusters[df_clusters.cluster == cluster]['text']
        tfidf_matrix = vectorizer.fit_transform(texts_in_cluster)
        mean_tfidf = np.mean(tfidf_matrix, axis=0)
        top_terms = np.array(vectorizer.get_feature_names_out())[np.argsort(mean_tfidf).flatten()[::-1][:10]]
        print(f"Cluster {cluster} top terms:", top_terms)

    # Sentiment analysis visualization
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_clusters, x='cluster', y='sentiment', palette='Set2')
    plt.title("Sentiment Distribution Across Clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Sentiment Score")
    plt.show()

    # Error analysis
    df_clusters['true_numeric'] = df_clusters['true_label'].map({'Terminal': 1, 'Non-Terminal': 0})
    misclassified = df_clusters[df_clusters['true_numeric'] != df_clusters['cluster']]
    print("Misclassified examples:", misclassified.head(10))

    df_clusters.to_csv("final_clustering_results.csv", index=False)
    logging.info("Final clustering analysis saved.")
