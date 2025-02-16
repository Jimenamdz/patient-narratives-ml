import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizer, DistilBertModel

# Download NLTK tokenizer if not already available
nltk.download('punkt')

# Load DistilBERT tokenizer & model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

def get_word_embedding(word, context):
    """Extract DistilBERT word embedding from a given context sentence."""
    if not word or not context:
        return None  # Instead of zero vector, return None
    
    inputs = tokenizer(context, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)

    tokenized_text = tokenizer.tokenize(context)

    # Find the word's token index (if present)
    try:
        word_index = tokenized_text.index(word.lower())  # Ensure lowercase match
        return outputs.last_hidden_state[:, word_index, :].numpy().flatten()
    except ValueError:
        return None  # If the word isn't in the sentence, return None

def load_data(file_path):
    """Load dataset and split into terminal vs. non-terminal patient texts."""
    df = pd.read_csv(file_path)
    terminal_texts = df[df["group"] == "terminal"]["text"].dropna().tolist()
    non_terminal_texts = df[df["group"] == "non-terminal"]["text"].dropna().tolist()
    return terminal_texts, non_terminal_texts

def compute_word_shifts(key_words, terminal_texts, non_terminal_texts):
    """Extract word embeddings and compute semantic shifts between groups."""
    terminal_embeddings = {word: [] for word in key_words}
    non_terminal_embeddings = {word: [] for word in key_words}

    for text in terminal_texts:
        for word in key_words:
            embedding = get_word_embedding(word, text)
            if np.any(embedding):  # Ignore zero vectors
                terminal_embeddings[word].append(embedding)

    for text in non_terminal_texts:
        for word in key_words:
            embedding = get_word_embedding(word, text)
            if np.any(embedding):
                non_terminal_embeddings[word].append(embedding)

    # Ensure we don't average empty lists
    avg_terminal_embeddings = {
        word: np.mean(embeds, axis=0) if embeds else np.zeros((768,))
        for word, embeds in terminal_embeddings.items()
    }
    avg_non_terminal_embeddings = {
        word: np.mean(embeds, axis=0) if embeds else np.zeros((768,))
        for word, embeds in non_terminal_embeddings.items()
    }

    return avg_terminal_embeddings, avg_non_terminal_embeddings

def compute_cosine_similarities(avg_terminal_embeddings, avg_non_terminal_embeddings):
    """Compute cosine similarity shift for key words, avoiding NaNs."""
    similarities = {}
    for word in avg_terminal_embeddings:
        if word in avg_non_terminal_embeddings:
            vec1 = avg_terminal_embeddings[word].reshape(1, -1)
            vec2 = avg_non_terminal_embeddings[word].reshape(1, -1)

            # Check if the vectors contain NaN values before computing similarity
            if np.isnan(vec1).any() or np.isnan(vec2).any():
                print(f"Warning: Skipping word '{word}' due to NaN values in embeddings.")
                continue  # Skip words with NaN values

            similarity = cosine_similarity(vec1, vec2)[0][0]
            similarities[word] = similarity

    return similarities

def visualize_word_shifts(avg_terminal_embeddings, avg_non_terminal_embeddings):
    """Visualize word shifts using UMAP"""
    word_labels = []
    all_embeddings = []
    word_groups = []

    for word, emb in avg_terminal_embeddings.items():
        if emb is not None:  # Ensure valid embeddings
            word_labels.append(f"{word} (Terminal)")
            all_embeddings.append(emb)
            word_groups.append(word)

    for word, emb in avg_non_terminal_embeddings.items():
        if emb is not None:  
            word_labels.append(f"{word} (Non-Terminal)")
            all_embeddings.append(emb)
            word_groups.append(word)

    all_embeddings = np.array(all_embeddings)

    if all_embeddings.shape[0] == 0:
        print("No valid word embeddings to visualize.")
        return

    umap_embeddings = umap.UMAP(n_components=2, random_state=42).fit_transform(all_embeddings)

    df_umap = pd.DataFrame(umap_embeddings, columns=["UMAP_1", "UMAP_2"])
    df_umap["Word"] = word_labels
    df_umap["Group"] = word_groups  

    plt.figure(figsize=(12, 8))

    scatter = sns.scatterplot(
        data=df_umap, x="UMAP_1", y="UMAP_2", hue="Group", palette="tab10",
        s=200, edgecolor="black", alpha=0.9
    )

    # **Ensures labels don't overlap**
    for i, txt in enumerate(df_umap["Word"]):
        plt.annotate(txt, 
                     (df_umap["UMAP_1"][i] + 0.15, df_umap["UMAP_2"][i] + 0.1), 
                     fontsize=10, alpha=0.8)

    plt.title("Semantic Shifts in Key Terms (Terminal vs. Non-Terminal)", fontsize=14)
    plt.xlabel("UMAP Dimension 1 (Semantic Difference 1)", fontsize=12)
    plt.ylabel("UMAP Dimension 2 (Semantic Difference 2)", fontsize=12)
    plt.legend(title="Key Term", loc="best", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.show()

if __name__ == "__main__":
    file_path = "patient_test_data.csv"  # Update when real data is ready
    key_words = ["hope", "treatment", "pain", "future", "scared", "relief"]

    try:
        # Load dataset
        terminal_texts, non_terminal_texts = load_data(file_path)

        # Compute word embeddings & shifts
        avg_terminal_embeddings, avg_non_terminal_embeddings = compute_word_shifts(key_words, terminal_texts, non_terminal_texts)
        cosine_similarities = compute_cosine_similarities(avg_terminal_embeddings, avg_non_terminal_embeddings)

        # Print cosine similarities
        print("\nCosine Similarity of Word Usage (Terminal vs. Non-Terminal):")
        for word, sim in cosine_similarities.items():
            print(f"{word}: {sim:.4f}")

        # Visualize word shifts
        visualize_word_shifts(avg_terminal_embeddings, avg_non_terminal_embeddings)

    except Exception as e:
        print(f"Error: {e}")