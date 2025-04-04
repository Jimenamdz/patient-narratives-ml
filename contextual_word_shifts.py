import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizerFast, DistilBertModel

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
device = torch.device("cpu")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

# Download NLTK tokenizer if not already available
nltk.download('punkt')


def get_word_embedding(word, context):
    inputs = tokenizer(context, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    word = word.lower()
    # Simple exact match, handles subwords correctly
    word_positions = [i for i, tok in enumerate(tokens) if tok.strip("#") == word]

    if not word_positions:
        return None

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[0, word_positions, :].cpu().numpy()
    word_embedding = np.mean(embeddings, axis=0)

    return word_embedding


def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['group', 'text', 'source']

    terminal_texts = df[df['group'].str.lower() == 'terminal']['text'].dropna().tolist()
    non_terminal_texts = df[df['group'].str.lower() == 'non-terminal']['text'].dropna().tolist()

    return terminal_texts, non_terminal_texts


def compute_word_shifts(key_words, terminal_texts, non_terminal_texts):
    terminal_embeddings = {word: [] for word in key_words}
    non_terminal_embeddings = {word: [] for word in key_words}

    for text in terminal_texts:
        for word in key_words:
            embedding = get_word_embedding(word, text)
            if embedding is not None:
                terminal_embeddings[word].append(embedding)

    for text in non_terminal_texts:
        for word in key_words:
            embedding = get_word_embedding(word, text)
            if embedding is not None:
                non_terminal_embeddings[word].append(embedding)

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
    similarities = {}
    for word in avg_terminal_embeddings:
        vec1 = avg_terminal_embeddings[word].reshape(1, -1)
        vec2 = avg_non_terminal_embeddings[word].reshape(1, -1)

        if np.isnan(vec1).any() or np.isnan(vec2).any():
            continue

        similarity = cosine_similarity(vec1, vec2)[0][0]
        similarities[word] = similarity

    return similarities


def visualize_word_shifts(avg_terminal_embeddings, avg_non_terminal_embeddings):
    word_labels, all_embeddings, word_groups = [], [], []

    for word, emb in avg_terminal_embeddings.items():
        word_labels.append(f"{word} (Terminal)")
        all_embeddings.append(emb)
        word_groups.append(word)

    for word, emb in avg_non_terminal_embeddings.items():
        word_labels.append(f"{word} (Non-Terminal)")
        all_embeddings.append(emb)
        word_groups.append(word)

    umap_embeddings = umap.UMAP(n_components=2, random_state=seed).fit_transform(np.array(all_embeddings))

    df_umap = pd.DataFrame(umap_embeddings, columns=["UMAP_1", "UMAP_2"])
    df_umap["Word"] = word_labels
    df_umap["Group"] = word_groups

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_umap, x="UMAP_1", y="UMAP_2", hue="Group", palette="tab10", s=200, edgecolor="black", alpha=0.9)

    for i, txt in enumerate(df_umap["Word"]):
        plt.annotate(txt, (df_umap["UMAP_1"][i] + 0.15, df_umap["UMAP_2"][i] + 0.1), fontsize=10, alpha=0.8)

    plt.title("Semantic Shifts in Key Terms (Terminal vs. Non-Terminal)", fontsize=14)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.legend(title="Key Term", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def plot_cosine_similarities(similarities):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(similarities.keys()), y=list(similarities.values()), palette="coolwarm")
    plt.title("Cosine Similarities (Terminal vs. Non-Terminal)")
    plt.ylabel("Cosine Similarity")
    plt.xlabel("Key Word")
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    file_path = "cleaned_patient_data.csv"
    key_words = ["hope", "treatment", "pain", "future", "scared", "relief"]

    try:
        terminal_texts, non_terminal_texts = load_data(file_path)

        avg_terminal_embeddings, avg_non_terminal_embeddings = compute_word_shifts(key_words, terminal_texts, non_terminal_texts)
        cosine_similarities = compute_cosine_similarities(avg_terminal_embeddings, avg_non_terminal_embeddings)

        print("\nCosine Similarity of Word Usage (Terminal vs. Non-Terminal):")
        for word, sim in cosine_similarities.items():
            print(f"{word}: {sim:.4f}")
            
        plot_cosine_similarities(cosine_similarities)
        

    except Exception as e:
        print(f"Error: {e}")
