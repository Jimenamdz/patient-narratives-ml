import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.decomposition import PCA

import os
import torch
import random
import numpy as np

# Set seeds and deterministic configurations
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

# Load DistilBERT tokenizer & model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

def get_sentence_embedding(text):
    """Extract sentence-level embedding using DistilBERT."""
    if not isinstance(text, str) or not text.strip():
        return np.zeros((768,))  # Return zero vector for empty or invalid text

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy().flatten()  # CLS token representation
    except Exception as e:
        print(f"Error processing text: {text[:30]}... -> {e}")
        return np.zeros((768,))  # Fallback to zero vector

def load_data(file_path):
    """Load dataset and split into terminal vs. non-terminal patient texts."""
    df = pd.read_csv(file_path)
    terminal_texts = df[df.iloc[:,0] == "terminal"].iloc[:,1].dropna().tolist()
    non_terminal_texts = df[df.iloc[:,0] == "non-terminal"].iloc[:,1].dropna().tolist()
    return terminal_texts, non_terminal_texts

def compute_embeddings(texts):
    """Compute embeddings for all texts and return as an array."""
    embeddings = [get_sentence_embedding(text) for text in texts if text.strip()]
    
    if not embeddings or all(np.all(emb == 0) for emb in embeddings):
        print("Warning: All embeddings are zero. Check input data.")
        return np.zeros((1, 768))  # Ensure valid shape

    return np.array(embeddings)

def permutation_test(terminal_embeddings, non_terminal_embeddings, n_permutations=1000):
    """Perform permutation testing to determine statistical significance of cosine distance."""
    try:
        if np.all(terminal_embeddings == 0) or np.all(non_terminal_embeddings == 0):
            print("Error: At least one group contains only zero embeddings.")
            return 0, 1.0, []  # Return neutral results

        observed_distance = cosine(terminal_embeddings.mean(axis=0), non_terminal_embeddings.mean(axis=0))
        combined = np.vstack([terminal_embeddings, non_terminal_embeddings])
        permuted_distances = []

        for _ in range(n_permutations):
            np.random.shuffle(combined)
            split_idx = len(terminal_embeddings)
            perm_terminal = combined[:split_idx].mean(axis=0)
            perm_non_terminal = combined[split_idx:].mean(axis=0)
            permuted_distances.append(cosine(perm_terminal, perm_non_terminal))

        p_value = (np.sum(np.array(permuted_distances) >= observed_distance) + 1) / (n_permutations + 1)
        return observed_distance, p_value, permuted_distances
    except Exception as e:
        print(f"Error during permutation test: {e}")
        return 0, 1.0, []  # Return neutral results

def plot_permutation_distribution(permuted_distances, observed_distance, p_value):
    """Plot the permutation test results to visualize significance with P-value annotation."""
    if not permuted_distances:
        print("Error: No valid permutation distances available for plotting.")
        return
    
    plt.figure(figsize=(10, 6))
    sns.histplot(permuted_distances, bins=50, kde=True, color="blue", alpha=0.7)
    plt.axvline(observed_distance, color="red", linestyle="dashed", label=f"Observed Distance: {observed_distance:.4f}")

    # Add P-value text to plot
    plt.text(
        x=min(permuted_distances), 
        y=max(plt.gca().get_ylim()) * 0.9, 
        s=f"P-value: {p_value:.4f}", 
        fontsize=12, 
        bbox=dict(facecolor="white", edgecolor="black")
    )

    plt.xlabel("Cosine Distance Between Groups")
    plt.ylabel("Frequency")
    plt.title("Permutation Test Distribution")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    file_path = "cleaned_patient_data.csv" 
    
    try:
        # Load dataset
        terminal_texts, non_terminal_texts = load_data(file_path)

        if not terminal_texts or not non_terminal_texts:
            print("Error: No valid text data found. Exiting...")
        else:
            # Compute all embeddings
            terminal_embeddings = compute_embeddings(terminal_texts)
            non_terminal_embeddings = compute_embeddings(non_terminal_texts)

            # --- Extra checks ---
            print(f"# Terminal Texts: {len(terminal_texts)}")
            print(f"# Non-Terminal Texts: {len(non_terminal_texts)}")
            print("Avg norm terminal embeddings:", np.mean(np.linalg.norm(terminal_embeddings, axis=1)))
            print("Avg norm non-terminal embeddings:", np.mean(np.linalg.norm(non_terminal_embeddings, axis=1)))

            # Check embedding shapes
            print(f"Terminal Embeddings Shape: {terminal_embeddings.shape}")
            print(f"Non-Terminal Embeddings Shape: {non_terminal_embeddings.shape}")

            # Run permutation testing
            observed_distance, p_value, permuted_distances = permutation_test(
                terminal_embeddings, non_terminal_embeddings
            )

            # Print results
            print(f"Observed Cosine Distance: {observed_distance:.4f}")
            print(f"P-value from Permutation Test: {p_value:.4f}")
            print(f"Mean permuted distance: {np.mean(permuted_distances):.4f}")
            print(f"95th percentile: {np.percentile(permuted_distances, 95):.4f}")
            print(f"Max permuted distance: {np.max(permuted_distances):.4f}")

            # Plot permutation test results with P-value
            plot_permutation_distribution(permuted_distances, observed_distance, p_value)

            # PCA visualization
            combined_embeddings = np.vstack([terminal_embeddings, non_terminal_embeddings])
            labels = ['terminal'] * len(terminal_embeddings) + ['non-terminal'] * len(non_terminal_embeddings)

            pca = PCA(n_components=2)
            proj = pca.fit_transform(combined_embeddings)

            plt.figure(figsize=(8, 6))
            plt.scatter(proj[:len(terminal_embeddings), 0], proj[:len(terminal_embeddings), 1],
                        alpha=0.5, label="Terminal")
            plt.scatter(proj[len(terminal_embeddings):, 0], proj[len(terminal_embeddings):, 1],
                        alpha=0.5, label="Non-Terminal")
            plt.title("PCA of Sentence Embeddings")
            plt.legend()
            plt.show()

    except Exception as e:
        print(f"Unexpected error: {e}")