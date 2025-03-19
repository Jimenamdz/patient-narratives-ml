import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizerFast, DistilBertModel

# Load DistilBERT tokenizer & model
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Download NLTK tokenizer if not already available
nltk.download('punkt')

def get_word_embedding(word, context):
    if not word or not context:
        return None
    
    inputs = tokenizer(context, return_tensors="pt", truncation=True, padding=True, max_length=128, return_offsets_mapping=True)
    
    with torch.no_grad():
        outputs = model(**{k: v for k, v in inputs.items() if k != "offset_mapping"})

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    offset_mapping = inputs['offset_mapping'][0]

    word = word.lower()
    word_positions = []

    for idx, (token, (start, end)) in enumerate(zip(tokens, offset_mapping)):
        token_clean = token.replace("##", "")
        token_text = context[start:end].lower()
        if word in token_text or token_clean == word:
            word_positions.append(idx)

    if not word_positions:
        return None  # Word not found after tokenization/truncation

    # Average embeddings for all sub-tokens of the word
    embeddings = outputs.last_hidden_state[0, word_positions, :].numpy()
    word_embedding = np.mean(embeddings, axis=0)
    
    return word_embedding

def load_data(file_path):
    """Load  dataset and split into terminal vs. non-terminal patient texts."""
    df = pd.read_csv(file_path)
    terminal_texts = df[df.iloc[:,0] == "terminal"].iloc[:,1].dropna().tolist()
    non_terminal_texts = df[df.iloc[:,0] == "non-terminal"].iloc[:,1].dropna().tolist()
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
    file_path = "cleaned_patient_data.csv"
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