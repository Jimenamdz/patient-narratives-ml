import torch
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer, DistilBertModel

# Load DistilBERT tokenizer & model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

def get_sentence_embedding(text):
    """Convert a sentence into a DistilBERT embedding."""
    if not text or not isinstance(text, str):  # Handle empty/non-string values
        return np.zeros((768,))
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()  # Extract [CLS] representation

def load_data(file_path):
    """Load real dataset and split into terminal vs. non-terminal patient texts."""
    df = pd.read_csv(file_path)
    terminal_texts = df[df.iloc[:,0] == "terminal"].iloc[:,1].dropna().tolist()
    non_terminal_texts = df[df.iloc[:,0] == "non-terminal"].iloc[:,1].dropna().tolist()
    return terminal_texts, non_terminal_texts

def generate_embeddings(text_list):
    """Compute DistilBERT embeddings for a list of sentences."""
    return np.array([get_sentence_embedding(text) for text in text_list])

def apply_umap(embeddings, n_components=2):
    """Reduce dimensionality of embeddings using UMAP."""
    reducer = umap.UMAP(n_components=n_components, n_neighbors=10, min_dist=0.2, random_state=42)
    return reducer.fit_transform(embeddings)

def visualize_clusters(terminal_embeddings, non_terminal_embeddings):
    """Plot UMAP visualization of terminal vs. non-terminal patient texts."""
    all_embeddings = np.vstack((terminal_embeddings, non_terminal_embeddings))
    umap_embeddings = apply_umap(all_embeddings)

    labels = (["Terminal"] * len(terminal_embeddings)) + (["Non-Terminal"] * len(non_terminal_embeddings))
    
    df_umap = pd.DataFrame(umap_embeddings, columns=["UMAP_1", "UMAP_2"])
    df_umap["Group"] = labels

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_umap, x="UMAP_1", y="UMAP_2", hue="Group",
        palette={"Terminal": "red", "Non-Terminal": "blue"}, 
        s=80, edgecolor="black", alpha=0.7
    )
    plt.title("UMAP Clustering of Patient Narratives", fontsize=14)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.legend(title="Group")
    plt.grid(True, linestyle="--", alpha=0.5)  # Add light grid for better readability
    plt.show()

if __name__ == "__main__":
    file_path = "cleaned_patient_data.csv" 

    try:
        # Load & embed data
        terminal_texts, non_terminal_texts = load_data(file_path)
        terminal_embeddings = generate_embeddings(terminal_texts)
        non_terminal_embeddings = generate_embeddings(non_terminal_texts)

        # Visualize clusters
        visualize_clusters(terminal_embeddings, non_terminal_embeddings)

    except Exception as e:
        print(f"Error: {e}")