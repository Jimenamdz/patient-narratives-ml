###############################################################################
#  IMPORTS
###############################################################################
import os
import re
import time
from datetime import datetime
import random
import praw
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from empath import Empath
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from collections import defaultdict, Counter
from tabulate import tabulate
import matplotlib.pyplot as plt
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')


###############################################################################
#  BASE CLASS
###############################################################################
class FeatureExtractor:
    def __init__(self, use_augmented=False, output_folder="data/feature_extracted_data"):
        """
        Initialize the extractor with support for augmented datasets.
        """
        self.use_augmented = use_augmented
        self.documents, self.labels = self.load_documents_and_labels()
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def load_documents_and_labels(self):
        """
        Load documents and globally unique labels for binary classification scenarios.
        """
        folder_base = "data/augmented_posts" if self.use_augmented else "data/preprocessed_posts"

        # Global class-to-label mapping
        label_map = {
            "non_terminal": 0,
            "terminal": 1,
        }

        # Binary classification folder mappings
        binary_folders = {
            "non_terminal_vs_terminal": ["non_terminal", "terminal"],
        }

        # Select the relevant pair for this execution
        selected_pair = "non_terminal_vs_terminal" 
        selected_classes = binary_folders[selected_pair]

        documents, labels = [], []
        for class_name in selected_classes:
            folder_path = f"{folder_base}/{class_name}"
            if not os.path.exists(folder_path):
                continue
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_name.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as file:
                        lines = file.readlines()
                        content = "".join(line.strip() for line in lines if not line.startswith(("Subreddit:", "Title:", "Author:", "Score:", "Created UTC:", "URL:"))).strip()
                        documents.append(content)
                        labels.append(label_map[class_name])
        return documents, labels


    def preprocess_text(self, text):
        """
        Tokenize, lowercase, remove stopwords, and stem.
        """
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        tokens = word_tokenize(text.lower())
        return [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]

    def save_to_csv(self, data, filename):
        """
        Save data to a CSV file.
        """
        # Construct the full file path
        if not filename.startswith(self.output_folder):
            file_path = os.path.join(self.output_folder, filename)
        else:
            file_path = filename

        # Debugging: Print the path being used
        print(f"Saving file to: {file_path}")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the file if it doesn't already exist
        if not os.path.exists(file_path):
            data.to_csv(file_path, index=False)
            print(f"Saved to {file_path}.")
        else:
            print(f"File already exists at {file_path}.")

###############################################################################
# N-GRAM FEATURE EXTRACTOR
###############################################################################
class NGramFeatureExtractor(FeatureExtractor):
    def __init__(self, use_augmented=False, output_folder="data/feature_extracted_data"):
        """
        Initialize NGramFeatureExtractor, optionally using augmented data.

        Args:
            use_augmented (bool): Whether to use augmented data for feature extraction.
            output_folder (str): Path to save the extracted feature files.
        """
        super().__init__(use_augmented=use_augmented, output_folder=output_folder)
        
        # Initialize vectorizers for unigrams, bigrams, and combined n-grams
        self.vectorizer_unigram = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', max_features=5000)
        self.vectorizer_bigram = TfidfVectorizer(ngram_range=(2, 2), stop_words='english', max_features=5000)
        self.vectorizer_combined = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=5000)
        
        # Placeholders for extracted features and feature names
        self.unigram_matrix = None
        self.bigram_matrix = None
        self.unigram_feature_names = None
        self.bigram_feature_names = None
        self.combined_matrix = None
        self.combined_feature_names = None

    def extract_features(self):
        """
        Extract unigram and bigram features using TF-IDF.
        """
        print("Extracting unigrams...")
        self.unigram_matrix = self.vectorizer_unigram.fit_transform(self.documents)
        self.unigram_feature_names = self.vectorizer_unigram.get_feature_names_out()
        print(f"Number of unigram features: {len(self.unigram_feature_names)}")

        print("Extracting bigrams...")
        self.bigram_matrix = self.vectorizer_bigram.fit_transform(self.documents)
        self.bigram_feature_names = self.vectorizer_bigram.get_feature_names_out()
        print(f"Number of bigram features: {len(self.bigram_feature_names)}")

        print("Extracting combined unigrams and bigrams...")
        self.combined_matrix = self.vectorizer_combined.fit_transform(self.documents)
        self.combined_feature_names = self.vectorizer_combined.get_feature_names_out()
        print(f"Number of combined unigram and bigram features: {len(self.combined_feature_names)}")

        return self.unigram_matrix, self.bigram_matrix, self.combined_matrix

    def save_features(self):
        """
        Save unigram and bigram features with labels to CSV files.
        Dynamically includes scenario and handles augmented data.
        """
        scenario = self.output_folder.split('/')[-1]  # Extract scenario from output folder path
        prefix = "augmented_" if self.use_augmented else ""

        # Define filenames with the scenario
        unigram_file = os.path.join(self.output_folder, f"{prefix}unigram_features_with_labels_{scenario}.csv")
        bigram_file = os.path.join(self.output_folder, f"{prefix}bigram_features_with_labels_{scenario}.csv")

        # Save unigram features
        if self.unigram_matrix is not None:
            unigram_df = pd.DataFrame(self.unigram_matrix.toarray(), columns=self.unigram_feature_names)
            unigram_df['label'] = self.labels
            unigram_df.to_csv(unigram_file, index=False)
            print(f"Saved unigram features to {unigram_file}.")
        else:
            print("No unigram features to save.")

        # Save bigram features
        if self.bigram_matrix is not None:
            bigram_df = pd.DataFrame(self.bigram_matrix.toarray(), columns=self.bigram_feature_names)
            bigram_df['label'] = self.labels
            bigram_df.to_csv(bigram_file, index=False)
            print(f"Saved bigram features to {bigram_file}.")
        else:
            print("No bigram features to save.")



    def compute_pmi(self, bigram, unigram_counts, total_bigrams, total_unigrams):
        """
        Compute the Pointwise Mutual Information (PMI) for a given bigram.
        """
        word1, word2 = bigram.split()
        p_bigram = total_bigrams[bigram] / sum(total_bigrams.values())
        p_word1 = unigram_counts[word1] / sum(total_unigrams.values())
        p_word2 = unigram_counts[word2] / sum(total_unigrams.values())
        return log(p_bigram / (p_word1 * p_word2), 2)

    def filter_bigrams_by_pmi(self, threshold=2.0):
        unigram_counts = Counter(word for doc in self.documents for word in doc.split())
        bigram_counts = Counter(f"{doc[i]}{doc[i+1]}" 
                                for doc in [d.split() for d in self.documents]
                                for i in range(len(doc)-1))
        valid_bigrams = {
            bigram: self.compute_pmi(bigram, unigram_counts, bigram_counts, unigram_counts)
            for bigram in self.bigram_feature_names
            if bigram in bigram_counts
        }
        self.bigram_feature_names = [
            bigram for bigram, pmi in valid_bigrams.items() if pmi >= threshold
        ]
        bigram_indices = [
            i for i, bigram in enumerate(self.vectorizer_bigram.get_feature_names_out())
            if bigram in self.bigram_feature_names
        ]
        self.bigram_matrix = self.bigram_matrix[:, bigram_indices]
    def compute_frequencies(self, feature_type="unigram"):
        """
        Compute frequencies of unigrams or bigrams for non-terminal and terminal posts.

        Parameters:
        feature_type (str): Specify 'unigram' or 'bigram' to compute respective frequencies.

        Returns:
        tuple: Two dictionaries containing frequencies for non-terminal and terminal posts.
        """
        if feature_type == "unigram":
            matrix = self.unigram_matrix
            feature_names = self.unigram_feature_names
        elif feature_type == "bigram":
            matrix = self.bigram_matrix
            feature_names = self.bigram_feature_names
        else:
            raise ValueError("Invalid feature_type. Choose 'unigram' or 'bigram'.")

        # Separate indices for non-terminal and terminal posts
        non_terminal_indices = [i for i, label in enumerate(self.labels) if label == 0]
        terminal_indices = [i for i, label in enumerate(self.labels) if label == 1]

        # Subset the matrix for each category
        non_terminal_matrix = matrix[non_terminal_indices]
        terminal_matrix = matrix[terminal_indices]

        # Sum the frequencies for each feature
        non_terminal_sums = non_terminal_matrix.sum(axis=0).A1
        terminal_sums = terminal_matrix.sum(axis=0).A1

        # Create frequency dictionaries
        non_terminal_freqs = {feature_names[i]: non_terminal_sums[i] for i in range(len(feature_names))}
        terminal_freqs = {feature_names[i]: terminal_sums[i] for i in range(len(feature_names))}

        return non_terminal_freqs, terminal_freqs
    
    def generate_wordclouds(self):
        """
        Generate word clouds for non-terminal and terminal groups based on unigrams and bigrams.
        """
        # Get unigram and bigram frequencies for all groups
        non_terminal_unigrams, terminal_unigrams = self.compute_frequencies(feature_type="unigram")
        non_terminal_bigrams, terminal_bigrams = self.compute_frequencies(feature_type="bigram")

        # Prepare word cloud data for each group and n-gram type
        wordcloud_data = {
            "Non-Terminal - Unigrams": non_terminal_unigrams,
            "Terminal - Unigrams": terminal_unigrams,
            "Non-Terminal - Bigrams": non_terminal_bigrams,
            "Terminal - Bigrams": terminal_bigrams,
        }

        # Dictionary to store generated word clouds
        generated_wordclouds = {}

        for title, frequencies in wordcloud_data.items():
            # Clean up frequencies by removing invalid or zero entries
            cleaned_frequencies = {word: freq for word, freq in frequencies.items() if not pd.isna(freq) and freq > 0}
            if not cleaned_frequencies:
                print(f"Skipping {title} due to empty or invalid frequency data.")
                continue

            # Generate word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(cleaned_frequencies)
            generated_wordclouds[title] = wordcloud  # Store the word cloud object

        return generated_wordclouds
    
    def save_wordclouds(self, output_folder):
        wordcloud_data = self.generate_wordclouds()
        os.makedirs(output_folder, exist_ok=True)
        for title, wordcloud in wordcloud_data.items():
            file_name = f"{title.replace(' ', '_').lower()}.png"
            file_path = os.path.join(output_folder, file_name)

            # Save the word cloud image
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(title, fontsize=16)
            plt.savefig(file_path, bbox_inches='tight', dpi=300)
            plt.close()  # Close the figure to avoid overlapping
            print(f"Saved word cloud: {file_path}")

###############################################################################
# EMPATH FEATURE-EXTRACTOR
###############################################################################

class EmpathFeatureExtractor(FeatureExtractor):
    def __init__(self, use_augmented=False, output_folder="data/feature_extracted_data"):
        """
        Initialize EmpathFeatureExtractor, optionally using augmented data.

        Args:
            use_augmented (bool): Whether to use augmented data for feature extraction.
            output_folder (str): Path to save the extracted feature files.
        """
        super().__init__(use_augmented=use_augmented, output_folder=output_folder)
        self.lexicon = Empath()
        self.features = None
        self.correlation_results = None
        self.significant_results = None

        # Categories to focus on (truncated for brevity)
        self.categories = {
            "linguistic_features": [
                "articles", "auxiliary_verbs", "adverbs", "conjunctions", 
                "personal_pronouns", "impersonal_pronouns", "negations", 
                "prepositions", "verbs", "nouns", "adjectives", 
                "comparatives", "superlatives", "modifiers", "function_words", 
                "filler_words", "verb_tense", "slang", "jargon", 
                "formal_language", "casual_language", "exclamations", 
                "contractions", "word_complexity", "sentiment_words"
            ],
                "psychological_processes": {
                "affective": [
                    "positive_emotion", "negative_emotion", "joy", "anger", 
                    "sadness", "anxiety", "fear", "disgust", "love", 
                    "hope", "trust", "excitement", "anticipation", 
                    "relief", "sympathy", "gratitude", "shame", 
                    "guilt", "envy", "pride", "contentment", "confusion",
                    "boredom", "embarrassment", "longing", "nostalgia", 
                    "embarrassment", "frustration", "surprise", "melancholy"
                ],
                "biological": [
                    "body", "health", "illness", "pain", "hygiene", 
                    "fitness", "exercise", "nutrition", "ingestion", 
                    "physical_state", "medicine", "sleep", "sexual", 
                    "aging", "disease", "injury", "hospital", "recovery", 
                    "dieting", "mental_health", "drug_use", "headache", 
                    "fatigue", "hormones", "appetite"
                ],
                "social": [
                    "family", "friends", "relationships", "group_behavior", 
                    "teamwork", "social_media", "communication", "community", 
                    "peer_pressure", "leadership", "parenting", "mentorship", 
                    "marriage", "divorce", "gender_roles", "social_identity", 
                    "cultural_rituals", "networking", "altruism", "conflict", 
                    "social_support", "dominance", "affiliation", "intimacy", 
                    "supportiveness", "competition", "conflict_resolution", 
                    "collaboration", "in-group", "out-group", "prejudice"
                ],
                "cognitive": [
                    "certainty", "doubt", "insight", "cause", "discrepancy", 
                    "problem_solving", "creativity", "self_reflection", "planning", 
                    "memory", "perception", "attention", "reasoning", "thought_process", 
                    "decision_making", "confusion", "learning", "metacognition", "adaptability", 
                    "focus", "perspective", "problem_analysis", "evaluation", "interpretation",
                    "logic", "intelligence", "rational_thought", "intuition", "conceptualization"
                ],
                "drives": [
                    "achievement", "dominance", "affiliation", "control", 
                    "self-esteem", "autonomy", "self-assertion", "power", 
                    "ambition", "conformity", "subordination", "dependence", 
                    "submission", "accomplishment", "independence", "order", 
                    "control_seeking", "status", "prosocial_behavior"
                ],
                "spiritual": [
                    "spirituality", "faith", "beliefs", "sacred", "religion", 
                    "prayer", "meditation", "afterlife", "soul", "divine", 
                    "god", "higher_power", "inspiration", "transcendence", 
                    "morality", "ethics", "rituals", "holiness", "mindfulness"
                ]
            },
            "personal_concerns": [
                "work", "money", "wealth", "shopping", "career", "travel", 
                "home", "school", "education", "violence", "death", 
                "retirement", "spirituality", "family_life", "hobbies", 
                "volunteering", "pets", "entertainment", "parenting", 
                "sports", "adventure", "politics", "environment", 
                "safety", "technology", "materialism", "status", 
                "self_improvement", "learning", "self_growth", "happiness", 
                "life_purpose", "work_life_balance", "stress", "coping", 
                "job_satisfaction", "ambition", "legacy", "job_search", 
                "unemployment", "retirement_plans", "mental_health", "dating", 
                "romantic_relationships", "divorce", "life_stressors", "transitions"
            ],
            "time_orientations": [
                "present", "past", "future", "morning", 
                "afternoon", "evening", "day", "night", 
                "weekdays", "weekends", "seasons", "holidays", 
                "lifespan", "long_term", "short_term", 
                "routine", "historical", "epoch", "momentary", 
                "timeliness", "timelessness", "urgency", 
                "progression", "nostalgia", "anticipation"
            ]
        }

    def extract_empath_features(self):
        """
        Extract empath features for each document.
        """
        features = []
        for doc in self.documents:
            doc_features = {}

            # Linguistic features
            for category in self.categories.get("linguistic_features", []):
                            doc_features[category] = self.lexicon.analyze(doc, categories=[category])[category]

            # Psychological processes
            for subcategory, subcategories in self.categories.get("psychological_processes", {}).items():
                for category in subcategories:
                    doc_features[category] = self.lexicon.analyze(doc, categories=[category])[category]

            # Personal concerns
            for category in self.categories.get("personal_concerns", []):
                doc_features[category] = self.lexicon.analyze(doc, categories=[category])[category]

            # Time orientations
            for category in self.categories.get("time_orientations", []):
                doc_features[category] = self.lexicon.analyze(doc, categories=[category])[category]

            features.append(doc_features)

        # Convert to a DataFrame
        self.features = pd.DataFrame(features)

       # Add labels to the features DataFrame
        if len(self.features) == len(self.labels):
            self.features['label'] = self.labels
            print("Added label column to the extracted features.")
        else:
            raise ValueError("Mismatch between the number of features and labels.")

        print(f"Extracted Empath features with shape: {self.features.shape}")

    def analyze_correlation(self):
        if self.features is None:
            raise ValueError("Features must be extracted before analyzing correlations.")

        # Remove constant columns
        constant_columns = self.features.columns[self.features.nunique() == 1]
        self.features.drop(columns=constant_columns, inplace=True, errors='ignore')
        print(f"Removed constant columns: {list(constant_columns)}")

        # Validate labels
        if len(set(self.labels)) == 1:
            raise ValueError("Labels array is constant; cannot compute correlation.")

        correlations, p_values = [], []

        for column in self.features.columns.drop("label"):
            correlation, p_value = pearsonr(self.features[column], self.labels)
            correlations.append(correlation)
            p_values.append(p_value)

        correction_results = multipletests(p_values, alpha=0.05, method="fdr_bh")
        _, corrected_p_values, _, _ = correction_results

        # Create a correlation DataFrame
        self.correlation_results = pd.DataFrame({
            "Feature": self.features.columns.drop("label"),
            "Correlation": correlations,
            "P-Value": p_values,
            "Corrected P-Value": corrected_p_values
        }).sort_values(by="Correlation", key=abs, ascending=False)

    def generate_correlation_table(self):
        if self.correlation_results is None:
            raise ValueError("Correlation results must be calculated before generating the table.")

        results = []

        for _, row in self.correlation_results.iterrows():
            feature = row["Feature"]
            correlation = row["Correlation"]
            p_value = row["P-Value"]

            # Look through each main category or subcategory
            for category, features in self.categories.items():
                if isinstance(features, dict):  # e.g. "psychological_processes"
                    for subcategory, subfeatures in features.items():
                        if feature in subfeatures:
                            results.append((f"{category} - {subcategory}", feature, correlation, p_value))
                elif feature in features:
                    results.append((category, feature, correlation, p_value))

        correlation_table = pd.DataFrame(results, columns=["Empath Category", "Example Word", "Correlation", "P-Value"])
        return correlation_table
    
    def save_correlation_table(self, output_folder):
        if self.correlation_results is None:
            raise ValueError("Correlation results must be calculated before saving.")
        
        correlation_table = self.generate_correlation_table()
        file_path = os.path.join(output_folder, "Empath_Correlation_Table.csv")
        correlation_table.to_csv(file_path, index=False)
        print(f"Saved Empath correlation table: {file_path}")

    def save_features_and_results(self, overwrite=False):
        """
        Save empath features with labels to a CSV file.
        Dynamically includes scenario and handles augmented data.
        """
        scenario = self.output_folder.split('/')[-1]  # Extract scenario from output folder path
        prefix = "augmented_" if self.use_augmented else ""
        feature_file = os.path.join(self.output_folder, f"{prefix}empath_features_with_labels_{scenario}.csv")

        # Save features if they exist and overwrite is not disabled
        if self.features is not None:
            if overwrite or not os.path.exists(feature_file):
                self.features.to_csv(feature_file, index=False)
                print(f"Saved empath features with labels to {feature_file}.")
            else:
                print(f"Empath features file already exists at {feature_file}.")
        else:
            print("No empath features to save.")



###############################################################################
#  LDA FEATURE EXTRACTOR
###############################################################################
class LDAFeatureExtractor(FeatureExtractor):
    def __init__(self, use_augmented=False, num_topics=70, passes=15, output_folder="data/feature_extracted_data", random_state=42):
        """
        Initialize LDAFeatureExtractor, optionally using augmented data.

        Args:
            use_augmented (bool): Whether to use augmented data for feature extraction.
            num_topics (int): Number of topics for LDA.
            passes (int): Number of passes for LDA training.
            output_folder (str): Path to save the extracted feature files.
            random_state (int): Random seed for reproducibility.
        """
        super().__init__(use_augmented=use_augmented, output_folder=output_folder)
        self.num_topics = num_topics
        self.passes = passes
        self.random_state = random_state
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.topic_distributions = None

    def preprocess_documents(self):
        """
        Preprocess documents: tokenize, remove stopwords, and stem.
        """
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        processed_docs = [
            [
                stemmer.stem(word) for word in word_tokenize(doc.lower())
                if word.isalpha() and word not in stop_words
            ]
            for doc in self.documents
        ]
        return processed_docs

    def filter_docs_by_word_count(self, processed_docs, min_documents=10):
        """
        Filter words that appear in more than 10 documents.
        """
        word_doc_count = defaultdict(int)
        for doc in processed_docs:
            unique_words = set(doc)
            for word in unique_words:
                word_doc_count[word] += 1

        filtered_docs = [
            [word for word in doc if word_doc_count[word] > min_documents]
            for doc in processed_docs
        ]
        return filtered_docs

    def train_lda(self, processed_docs):
        """
        Train the LDA model.
        """
        self.dictionary = corpora.Dictionary(processed_docs)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        self.lda_model = LdaModel(
            self.corpus, num_topics=self.num_topics, id2word=self.dictionary, passes=self.passes,
            random_state=self.random_state
        )

    def extract_topic_distributions(self):
        """
        Extract topic distributions for each document.
        """
        self.topic_distributions = [
            dict(self.lda_model.get_document_topics(doc, minimum_probability=0))
            for doc in self.corpus
        ]
    

    def topic_distribution_to_matrix(self):
        """
        Convert topic distributions to a matrix format.
        """
        matrix = np.zeros((len(self.topic_distributions), self.num_topics))
        for i, distribution in enumerate(self.topic_distributions):
            for topic_id, prob in distribution.items():
                matrix[i, topic_id] = prob
        return matrix

    def save_features(self):
        """
        Save LDA topic distributions with labels to a CSV file.
        Dynamically includes scenario and handles augmented data.
        """
        if not self.topic_distributions:
            raise ValueError("Topic distributions are not extracted.")
        
        # Extract the scenario from the output folder path
        scenario = self.output_folder.split('/')[-1]  # Adjust for your folder structure
        prefix = "augmented_" if self.use_augmented else ""
        
        # Define the output file path
        lda_features_file = os.path.join(self.output_folder, f"{prefix}lda_topic_distributions_with_labels_{scenario}.csv")
        
        # Save the features file
        if not os.path.exists(lda_features_file):
            # Convert topic distributions to a matrix format
            topic_matrix = self.topic_distribution_to_matrix()
            labels_array = np.array(self.labels)
            
            # Create a DataFrame with the topic distributions and labels
            lda_features_df = pd.DataFrame(topic_matrix)
            lda_features_df['label'] = labels_array
            lda_features_df.to_csv(lda_features_file, index=False)
            
            print(f"Saved LDA topic distributions with labels to {lda_features_file}.")
        else:
            print(f"LDA features file already exists at {lda_features_file}.")

    def run_extraction_pipeline(self):
        """
        Complete LDA pipeline: preprocess, train, extract, visualize, and save.
        """
        print("Preprocessing documents...")
        processed_docs = self.preprocess_documents()
        filtered_docs = self.filter_docs_by_word_count(processed_docs)

        print("Training LDA model...")
        self.train_lda(filtered_docs)

        print("Extracting topic distributions...")
        self.extract_topic_distributions()

        print("Saving features...")
        self.save_features()

        print("LDA pipeline complete.")

    def run_analysis_pipeline(self):
        """
        Complete LDA analysis pipeline with t-SNE visualization.
        """
        print("Starting LDA analysis pipeline...")

        # Preprocess
        print("Preprocessing documents...")
        processed_docs = self.preprocess_documents()
        filtered_docs = self.filter_docs_by_word_count(processed_docs)

        # Train the LDA model
        print("Training LDA model...")
        self.train_lda(filtered_docs)

        # Extract topic distributions
        print("Extracting topic distributions...")
        self.extract_topic_distributions()

        # Convert to matrix
        print("Converting topic distributions to matrix format...")
        self.topic_matrix = self.topic_distribution_to_matrix()

        print("LDA analysis pipeline complete.")


###############################################################################
#  SUMMARY / TABLE GENERATION FUNCTIONS
###############################################################################

# Creating a summary table for the number of features extracted
def generate_summary_table(ngram_extractor, empath_extractor, lda_extractor, output_file=None):
    # Extract the number of features from each extractor
    unigram_count = len(ngram_extractor.unigram_feature_names)
    bigram_count = len(ngram_extractor.bigram_feature_names)
    empath_feature_count = empath_extractor.features.shape[1] - 1  # Exclude label column
    lda_feature_count = lda_extractor.num_topics  # Number of topics in LDA

    # Build the summary data
    summary_data = [
        ["N-grams", "Unigram", unigram_count],
        ["N-grams", "Bigram", bigram_count],
        ["Linguistic Dimensions", "Empath", empath_feature_count],
        ["Topic Modeling", "LDA", lda_feature_count]
    ]

    # Convert to a DataFrame
    summary_table = pd.DataFrame(summary_data, columns=["Feature Type", "Methods", "Number of Selected Features"])

    # Plot the table
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    plt.title("Summary of Feature Extraction Methods", fontsize=16, fontweight='bold', pad=20)

    # Create the table 
    table = ax.table(cellText=summary_table.values, 
                     colLabels=summary_table.columns, 
                     cellLoc='center', 
                     loc='center')

    # Style adjustments
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1, 2])  
    table.scale(1.2, 1.2)  
    for (row, col), cell in table.get_celld().items():
        if row == 0: 
            cell.set_fontsize(12)
            cell.set_text_props(weight="bold")
            cell.set_linewidth(1.5)
        if col == 0:  
            cell.set_text_props(weight="bold")
        if row > 0 and (row % 2 == 0):  
            cell.set_facecolor("#f0f0f0")

    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    print(f"Table saved to {output_file}")
    plt.show()

    return summary_table

# Creating a summary table for the correlations from the EMPATH features extracted
def generate_empath_table(input_csv, output_file=None):
    empath_df = pd.read_csv(input_csv)

    # Sort by p-value and filter the top features for display
    empath_df["Empath Category"] = empath_df["Empath Category"].str.capitalize()
    empath_df = empath_df.sort_values(by="P-Value").head(10).sort_values(by="Empath Category")
    empath_df = empath_df.drop(columns=["Correlation"])

    empath_df.rename(
        columns={"Empath Category": "Category", "Example word": "Example Word", "P value": "P-Value"}, inplace=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=empath_df.values,
        colLabels=empath_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(empath_df.columns))))
    for key, cell in table.get_celld().items():
        row, col = key
        if row == 0: 
            cell.set_text_props(weight="bold")
            cell.set_fontsize(14)



    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    print(f"Table saved to {output_file}")
    plt.show()