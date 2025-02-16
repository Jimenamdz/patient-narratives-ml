from feature_extraction_func import NGramFeatureExtractor, EmpathFeatureExtractor, LDAFeatureExtractor

# Define all binary classification scenarios
scenarios = {
    "non_terminal_vs_terminal": ["non_terminal", "terminal"]
}

# Process features for each scenario
for scenario in scenarios.keys():
    print(f"\nProcessing features for {scenario}...")

    # Define output folder for the scenario
    output_folder = f"data/feature_extracted_data/{scenario}"

    # Process Original Data
    print(f"Processing original data for {scenario}...")
    ngram_extractor = NGramFeatureExtractor(output_folder=output_folder)
    ngram_extractor.documents, ngram_extractor.labels = ngram_extractor.load_documents_and_labels()
    ngram_extractor.extract_features()
    ngram_extractor.save_features()

    empath_extractor = EmpathFeatureExtractor(output_folder=output_folder)
    empath_extractor.documents, empath_extractor.labels = empath_extractor.load_documents_and_labels()
    empath_extractor.extract_empath_features()
    empath_extractor.analyze_correlation()
    empath_extractor.save_features_and_results()

    lda_extractor = LDAFeatureExtractor(output_folder=output_folder)
    lda_extractor.documents, lda_extractor.labels = lda_extractor.load_documents_and_labels()
    lda_extractor.run_extraction_pipeline()

    # Process Augmented Data
    print(f"Processing augmented data for {scenario}...")
    ngram_extractor_augmented = NGramFeatureExtractor(use_augmented=True, output_folder=output_folder)
    ngram_extractor_augmented.documents, ngram_extractor_augmented.labels = ngram_extractor_augmented.load_documents_and_labels()
    ngram_extractor_augmented.extract_features()
    ngram_extractor_augmented.save_features()

    empath_extractor_augmented = EmpathFeatureExtractor(use_augmented=True, output_folder=output_folder)
    empath_extractor_augmented.documents, empath_extractor_augmented.labels = empath_extractor_augmented.load_documents_and_labels()
    empath_extractor_augmented.extract_empath_features()
    empath_extractor_augmented.analyze_correlation()
    empath_extractor_augmented.save_features_and_results()

    lda_extractor_augmented = LDAFeatureExtractor(use_augmented=True, output_folder=output_folder)
    lda_extractor_augmented.documents, lda_extractor_augmented.labels = lda_extractor_augmented.load_documents_and_labels()
    lda_extractor_augmented.run_extraction_pipeline()

print("\nFeature extraction completed for all scenarios.")
