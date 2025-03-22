import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

def get_sentence_embedding(text):
    """Convert a sentence into a DistilBERT embedding."""
    if not text or not isinstance(text, str):  # Handle empty or non-string values
        return np.zeros((768,))  # Return a zero-vector of appropriate size

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()  # Extract [CLS] token representation

def compute_similarity(terminal_texts, non_terminal_texts):
    """
    Compute cosine similarity between embeddings of two groups (Terminal vs Non-Terminal).
    
    Parameters:
    - terminal_texts (list): List of text entries from terminally ill patients.
    - non_terminal_texts (list): List of text entries from non-terminal patients.

    Returns:
    - Cosine similarity matrix (NumPy array)
    """
    terminal_embeddings = np.array([get_sentence_embedding(text) for text in terminal_texts])
    non_terminal_embeddings = np.array([get_sentence_embedding(text) for text in non_terminal_texts])

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(terminal_embeddings, non_terminal_embeddings)

    return similarity_matrix

def load_data(file_path):
    """Load data from cleaned CSV file."""
    df = pd.read_csv(file_path)

    df.columns = ['group', 'text', 'source']

    terminal_texts = df[df['group'].str.lower() == 'terminal']['text'].dropna().tolist()
    non_terminal_texts = df[df['group'].str.lower() == 'non-terminal']['text'].dropna().tolist()

    return terminal_texts, non_terminal_texts

if __name__ == "__main__":
    file_path = "cleaned_patient_data.csv" 
    try:
        terminal_texts, non_terminal_texts = load_data(file_path)
        similarity_matrix = compute_similarity(terminal_texts, non_terminal_texts)

        print("Cosine Similarity Matrix:")
        print(similarity_matrix)
    except Exception as e:
        print(f"Error: {e}")