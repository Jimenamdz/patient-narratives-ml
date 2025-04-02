import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    calinski_harabasz_score, davies_bouldin_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# === Logging setup ===
logging.basicConfig(filename='clustering_analysis.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')
logging.info("Enhanced clustering analysis started.")

# === Reproducibility ===
seed = 42
np.random.seed(seed)

# === Constants ===
FILE_PATH = "cleaned_patient_data.csv"

# === Load Data ===
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['group', 'text', 'source']
    df.dropna(subset=['text'], inplace=True)
    df['label'] = df['group'].str.lower().map({'terminal': 1, 'non-terminal': 0})
    return df

data_df = load_data(FILE_PATH)
texts = data_df['text'].tolist()
labels_true = data_df['label'].values

# === Generate Sentence Embeddings (SentenceTransformer) ===
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)

# === Sentiment Analysis ===
analyzer = SentimentIntensityAnalyzer()
sentiments = np.array([analyzer.polarity_scores(text)['compound'] for text in texts])

# === Combine embeddings and sentiment scores ===
combined_features = np.hstack((embeddings, sentiments.reshape(-1, 1)))

# === Optimal cluster number (Silhouette analysis) ===
print("Determining optimal number of clusters...")
silhouette_scores = []
k_range = range(2, 7)
for k in tqdm(k_range, desc="Evaluating number of clusters"):
    clusterer = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
    labels_temp = clusterer.fit_predict(combined_features)
    score = silhouette_score(combined_features, labels_temp)
    silhouette_scores.append(score)

# Plot optimal number of clusters
plt.figure(figsize=(8, 5))
plt.plot(k_range, silhouette_scores, 'o-', color='blue')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Optimal Number of Clusters')
plt.grid(True)
plt.show()

optimal_k = k_range[np.argmax(silhouette_scores)]
logging.info(f"Optimal number of clusters selected: {optimal_k}")

# === Agglomerative Clustering (cosine similarity) ===
print(f"Running Agglomerative clustering with {optimal_k} clusters...")
agg_cluster = AgglomerativeClustering(n_clusters=optimal_k, affinity='cosine', linkage='average')
cluster_labels = agg_cluster.fit_predict(combined_features)

# === Evaluation Metrics ===
silhouette_avg = silhouette_score(combined_features, cluster_labels)
ari_score = adjusted_rand_score(labels_true, cluster_labels)
ch_score = calinski_harabasz_score(combined_features, cluster_labels)
db_score = davies_bouldin_score(combined_features, cluster_labels)

print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
print(f"Calinski-Harabasz Index: {ch_score:.2f}")
print(f"Davies-Bouldin Index: {db_score:.2f}")

logging.info(
    f"Evaluation metrics - Silhouette: {silhouette_avg:.4f}, ARI: {ari_score:.4f}, "
    f"Calinski-Harabasz: {ch_score:.2f}, Davies-Bouldin: {db_score:.2f}")

# === TF-IDF Top Terms per Cluster ===
print("Performing TF-IDF analysis per cluster...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
df_clusters = pd.DataFrame({
    'text': texts,
    'cluster': cluster_labels,
    'sentiment': sentiments,
    'true_label': labels_true
})

for cluster in range(optimal_k):
    texts_cluster = df_clusters[df_clusters.cluster == cluster]['text']
    tfidf_matrix = vectorizer.fit_transform(texts_cluster)
    mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = mean_tfidf.argsort()[::-1][:10]
    top_terms = np.array(vectorizer.get_feature_names_out())[top_indices]
    print(f"Cluster {cluster} top terms:", top_terms)

# === Sentiment Distribution per Cluster ===
plt.figure(figsize=(8, 6))
sns.boxplot(x='cluster', y='sentiment', data=df_clusters, palette='Set3')
plt.title("Sentiment Distribution by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Sentiment Score")
plt.show()

# === Misclassification Analysis ===
misclassified = df_clusters[df_clusters['true_label'] != df_clusters['cluster']]
print(f"Number of misclassified examples: {len(misclassified)}")
print("Examples of misclassified texts:")
print(misclassified[['text', 'true_label', 'cluster']].head(10))

# === Save Results ===
df_clusters.to_csv("agglomerative_clustering_results.csv", index=False)
logging.info("Final enhanced clustering analysis saved successfully.")
