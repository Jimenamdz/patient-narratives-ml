import torch
import random
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

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

# Load DistilBERT tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

def get_sentence_embedding(text):
    """Convert a sentence into a DistilBERT embedding."""
    if not text or not isinstance(text, str):
        return np.zeros((768,))

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

def compute_similarity(terminal_texts, non_terminal_texts):
    """Compute cosine similarity between embeddings of two groups."""
    terminal_embeddings = np.array([get_sentence_embedding(text) for text in terminal_texts])
    non_terminal_embeddings = np.array([get_sentence_embedding(text) for text in non_terminal_texts])

    similarity_matrix = cosine_similarity(terminal_embeddings, non_terminal_embeddings)
    return similarity_matrix

def load_data(file_path):
    """Load data from cleaned CSV file."""
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
        similarity_matrix = compute_similarity(terminal_texts, non_terminal_texts)

        print("Cosine Similarity Matrix:")
        print(similarity_matrix)
    except Exception as e:
        print(f"Error: {e}")
