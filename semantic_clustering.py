import os
import torch
import random
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizerFast, DistilBertModel
from tqdm import tqdm

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

# Check GPU availability (temporarily forced to CPU for reproducibility testing)
device = torch.device("cpu")

# Constants (easily adjustable)
BATCH_SIZE = 32
MAX_LENGTH = 128
FILE_PATH = "cleaned_patient_data.csv"

# Initialize tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)


def get_sentence_embeddings(texts, batch_size=BATCH_SIZE):
    """Compute sentence embeddings using mean pooling in batches for efficiency."""
    embeddings = []
    model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding sentences"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", truncation=True,
            padding=True, max_length=MAX_LENGTH
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        batch_embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)

    return np.array(embeddings)


def load_data(file_path):
    """Load dataset and return separated lists for terminal/non-terminal texts."""
    df = pd.read_csv(file_path)
    df.columns = ['group', 'text', 'source']

    terminal_texts = df[df['group'].str.lower() == 'terminal']['text'].dropna().tolist()
    non_terminal_texts = df[df['group'].str.lower() == 'non-terminal']['text'].dropna().tolist()
    return terminal_texts, non_terminal_texts


def apply_umap(embeddings, n_components=2, n_neighbors=10, min_dist=0.2):
    """Reduce embeddings dimensionality using UMAP with reproducible settings."""
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed
    )
    return reducer.fit_transform(embeddings)


def visualize_embeddings(umap_embeddings, labels):
    """Visualize UMAP embeddings with clear plotting style."""
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
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    try:
        print("Loading data...")
        terminal_texts, non_terminal_texts = load_data(FILE_PATH)

        print("Generating embeddings...")
        terminal_embeddings = get_sentence_embeddings(terminal_texts)
        non_terminal_embeddings = get_sentence_embeddings(non_terminal_texts)

        print("Applying UMAP...")
        all_embeddings = np.vstack((terminal_embeddings, non_terminal_embeddings))
        umap_embeddings = apply_umap(all_embeddings)

        labels = (["Terminal"] * len(terminal_embeddings)) + (["Non-Terminal"] * len(non_terminal_embeddings))

        print("Visualizing embeddings...")
        visualize_embeddings(umap_embeddings, labels)

    except Exception as e:
        print(f"An error occurred: {e}")