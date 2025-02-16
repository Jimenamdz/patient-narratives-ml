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
    """Load data from a CSV or JSON file."""
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or JSON.")

    # Expecting columns: 'text' and 'group' (where group is 'terminal' or 'non-terminal')
    terminal_texts = df[df['group'] == 'terminal']['text'].dropna().tolist()
    non_terminal_texts = df[df['group'] == 'non-terminal']['text'].dropna().tolist()

    return terminal_texts, non_terminal_texts

def visualize_similarity(similarity_matrix):
    """Plot the cosine similarity matrix as a heatmap with improved visualization."""
    plt.figure(figsize=(10, 8))  # Increase figure size for better readability
    
    # Set a better color range to highlight differences
    vmin, vmax = 0.85, 1.0  # Adjust color scale to emphasize meaningful differences
    
    ax = sns.heatmap(similarity_matrix, 
                      annot=False,  # Remove numbers to declutter the heatmap
                      cmap="coolwarm", 
                      fmt=".2f", 
                      linewidths=0.5,  # Add gridlines for separation
                      vmin=vmin, vmax=vmax)  # Define color scale
    
    plt.xlabel("Non-Terminal Posts")
    plt.ylabel("Terminal Posts")
    plt.title("Cosine Similarity Between Terminal and Non-Terminal Patient Texts")
    plt.show()

if __name__ == "__main__":
    # Example usage with a real dataset
    file_path = "patient_test_data.csv"  # Change to your actual data file
    try:
        terminal_texts, non_terminal_texts = load_data(file_path)
        similarity_matrix = compute_similarity(terminal_texts, non_terminal_texts)
        
        print("Cosine Similarity Matrix:")
        print(similarity_matrix)
        
        visualize_similarity(similarity_matrix)  # Display heatmap
    except Exception as e:
        print(f"Error: {e}")

