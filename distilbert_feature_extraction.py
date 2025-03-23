import os
import torch
import random
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Set seeds and deterministic configurations for reproducibility
seed = 42
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(seed)

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Load DistilBERT tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)


def get_sentence_embedding(text):
    """Extract sentence-level embedding using mean pooling of token embeddings."""
    if not text or not isinstance(text, str) or not text.strip():
        return np.zeros((768,))

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state
    sentence_embedding = torch.mean(token_embeddings, dim=1).cpu().numpy().flatten()

    return sentence_embedding


def compute_similarity(terminal_texts, non_terminal_texts):
    """Compute cosine similarity between embeddings of two groups."""
    terminal_embeddings = np.array([get_sentence_embedding(text) for text in terminal_texts if text.strip()])
    non_terminal_embeddings = np.array([get_sentence_embedding(text) for text in non_terminal_texts if text.strip()])

    similarity_matrix = cosine_similarity(terminal_embeddings, non_terminal_embeddings)
    return similarity_matrix


def load_data(file_path):
    """Load data from cleaned CSV file and filter empty texts."""
    df = pd.read_csv(file_path)
    df.columns = ['group', 'text', 'source']

    terminal_texts = df[df['group'].str.lower() == 'terminal']['text'].dropna().tolist()
    non_terminal_texts = df[df['group'].str.lower() == 'non-terminal']['text'].dropna().tolist()

    return terminal_texts, non_terminal_texts


def visualize_similarity(similarity_matrix):
    plt.figure(figsize=(10, 7))
    sns.heatmap(similarity_matrix, cmap="coolwarm", annot=False)
    plt.title("Cosine Similarity Matrix between Terminal and Non-Terminal Texts")
    plt.xlabel("Non-Terminal Texts")
    plt.ylabel("Terminal Texts")
    plt.show()

if __name__ == "__main__":
    file_path = "cleaned_patient_data.csv"

    try:
        terminal_texts, non_terminal_texts = load_data(file_path)
        
        # Compute embeddings explicitly
        terminal_embeddings = np.array([get_sentence_embedding(text) for text in terminal_texts if text.strip()])
        non_terminal_embeddings = np.array([get_sentence_embedding(text) for text in non_terminal_texts if text.strip()])

        # Save embeddings for reuse
        np.save("terminal_embeddings.npy", terminal_embeddings)
        np.save("non_terminal_embeddings.npy", non_terminal_embeddings)
        print("Embeddings saved successfully as 'terminal_embeddings.npy' and 'non_terminal_embeddings.npy'")

        # Compute and save cosine similarity matrix
        similarity_matrix = compute_similarity(terminal_texts, non_terminal_texts)
        np.save("cosine_similarity_matrix.npy", similarity_matrix)
        print("Cosine similarity matrix saved as 'cosine_similarity_matrix.npy'")

        # Visualize similarity
        visualize_similarity(similarity_matrix)

    except Exception as e:
        print(f"Error: {e}")