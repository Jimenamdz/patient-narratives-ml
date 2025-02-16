###############################################################################
#  IMPORTS
###############################################################################
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../feature_extraction"))
from feature_extraction_func import NGramFeatureExtractor, LDAFeatureExtractor, EmpathFeatureExtractor, generate_summary_table, generate_empath_table # type: ignore

output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

# N-Gram Analysis
ngram_extractor = NGramFeatureExtractor()
ngram_extractor.extract_features()
ngram_extractor.save_wordclouds(output_folder)

# Empath Analysis
empath_extractor = EmpathFeatureExtractor()
empath_extractor.extract_empath_features()
empath_extractor.analyze_correlation()
empath_extractor.save_correlation_table(output_folder)

# LDA Analysis
lda_extractor = LDAFeatureExtractor(num_topics=20)
lda_extractor.run_analysis_pipeline()

# Getting the output table for number of features found in every extraction
generate_summary_table(ngram_extractor, empath_extractor, lda_extractor, f"{output_folder}/summary_table")

# Getting the output table for EMPATH feature correlations
generate_empath_table("outputs/Empath_Correlation_Table.csv", f"{output_folder}/empath_table")

def compare_features():
    original_extractor = NGramFeatureExtractor(use_augmented=False)
    augmented_extractor = NGramFeatureExtractor(use_augmented=True)

    original_features = original_extractor.extract_features()
    augmented_features = augmented_extractor.extract_features()

    # Compare feature counts or distributions
    print(f"Original Unigram Features: {len(original_extractor.unigram_feature_names)}")
    print(f"Augmented Unigram Features: {len(augmented_extractor.unigram_feature_names)}")

    # Visualize the differences
    original_extractor.save_wordclouds("outputs/original")
    augmented_extractor.save_wordclouds("outputs/augmented")