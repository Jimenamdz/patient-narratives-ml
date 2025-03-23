import os
import torch
import random
import numpy as np
import pandas as pd
from bertopic import BERTopic
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt

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


def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = ["group", "text", "source"]
    df = df.dropna(subset=["text"])
    return df


def run_topic_modeling(texts, group_name):
    topic_model = BERTopic(language="english", verbose=True)
    topics, probs = topic_model.fit_transform(texts)
    topic_model.save(f"{group_name}_bertopic_model")
    return topic_model


def analyze_sentiments(texts):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(text)["compound"] for text in texts if text.strip()]
    return sentiments


def plot_sentiments(terminal_sentiments, non_terminal_sentiments):
    plt.figure(figsize=(10, 6))
    sns.histplot(terminal_sentiments, color="red", label="Terminal", kde=True)
    sns.histplot(non_terminal_sentiments, color="blue", label="Non-Terminal", kde=True)
    plt.legend()
    plt.title("Sentiment Distribution: Terminal vs Non-Terminal")
    plt.xlabel("Sentiment (VADER compound)")
    plt.ylabel("Frequency")
    plt.savefig("sentiment_distribution.png")
    plt.show()


def print_top_keywords(topic_model, group_name, num_topics=5):
    print(f"\n{group_name} Top Keywords:")
    for i in range(num_topics):
        topic_words = topic_model.get_topic(i)
        print(f"Topic {i}: {topic_words}")


if __name__ == "__main__":
    file_path = "cleaned_patient_data.csv"
    data_df = load_data(file_path)

    terminal_texts = data_df[data_df["group"].str.lower() == "terminal"]["text"].tolist()
    non_terminal_texts = data_df[data_df["group"].str.lower() == "non-terminal"]["text"].tolist()

    print(f"# Terminal Texts: {len(terminal_texts)}")
    print(f"# Non-Terminal Texts: {len(non_terminal_texts)}")

    # Topic modeling (combined for direct comparison)
    combined_texts = terminal_texts + non_terminal_texts
    combined_model = run_topic_modeling(combined_texts, "combined")

    # Print topics
    print_top_keywords(combined_model, "Combined Group Topics")

    # Sentiment analysis
    terminal_sentiments = analyze_sentiments(terminal_texts)
    non_terminal_sentiments = analyze_sentiments(non_terminal_texts)

    # Debugging sentiment analysis output
    print(f"Mean Terminal Sentiment: {np.mean(terminal_sentiments):.4f}")
    print(f"Mean Non-Terminal Sentiment: {np.mean(non_terminal_sentiments):.4f}")

    # Sentiment visualization
    plot_sentiments(terminal_sentiments, non_terminal_sentiments)
