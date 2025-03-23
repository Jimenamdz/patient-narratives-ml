import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# ============================
# Feature Preparation
# ============================

# Load terminal and non-terminal embeddings (ensure these files are created by earlier scripts)
terminal_embeddings = np.load("terminal_embeddings.npy")  # saved from earlier steps
non_terminal_embeddings = np.load("non_terminal_embeddings.npy")  # saved from earlier steps

# Load sentiment scores (you should save sentiment arrays in previous sentiment analysis step)
terminal_sentiment = np.load("terminal_sentiment.npy")
non_terminal_sentiment = np.load("non_terminal_sentiment.npy")

# Combine embeddings and sentiments into DataFrames
terminal_df = pd.DataFrame(terminal_embeddings)
terminal_df['sentiment'] = terminal_sentiment
terminal_df['label'] = 1  # Terminal group labeled as 1

non_terminal_df = pd.DataFrame(non_terminal_embeddings)
non_terminal_df['sentiment'] = non_terminal_sentiment
non_terminal_df['label'] = 0  # Non-terminal group labeled as 0

# Merge data into single dataset
full_df = pd.concat([terminal_df, non_terminal_df], ignore_index=True)
full_df.to_csv("features_and_sentiment.csv", index=False)
print("Step 1 complete: Combined features saved to 'features_and_sentiment.csv'")

# ================================
# Machine Learning Modeling
# ================================

# Split dataset
X = full_df.drop('label', axis=1)
y = full_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Train Logistic Regression (interpretable baseline)
logreg = LogisticRegression(max_iter=500, random_state=42)
logreg.fit(X_train, y_train)
logreg_preds = logreg.predict(X_test)

# Train Random Forest (captures more complexity)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Save trained models
joblib.dump(logreg, "logistic_regression.pkl")
joblib.dump(rf, "random_forest.pkl")
print("Step 2 complete: Models trained and saved.")

# Evaluate models
print("\n--- Logistic Regression Performance ---")
print(classification_report(y_test, logreg_preds, target_names=['Non-Terminal', 'Terminal']))
print("Confusion Matrix:\n", confusion_matrix(y_test, logreg_preds))

print("\n--- Random Forest Performance ---")
print(classification_report(y_test, rf_preds, target_names=['Non-Terminal', 'Terminal']))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_preds))

# Save reports
with open("ml_classification_report.txt", "w") as f:
    f.write("Logistic Regression Report\n")
    f.write(classification_report(y_test, logreg_preds, target_names=['Non-Terminal', 'Terminal']))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, logreg_preds)))

    f.write("\n\nRandom Forest Report\n")
    f.write(classification_report(y_test, rf_preds, target_names=['Non-Terminal', 'Terminal']))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, rf_preds)))

# =================================
# Feature Importance Analysis
# =================================

# Permutation importance for Random Forest
perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=42)

# Extract importance scores
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": perm_importance.importances_mean,
    "importance_std": perm_importance.importances_std
}).sort_values(by="importance_mean", ascending=False)

# Identify importance of sentiment explicitly
sentiment_importance = importance_df[importance_df["feature"] == 'sentiment'].iloc[0]

print("\nFeature Importance Analysis:")
print(f"Sentiment Importance: mean={sentiment_importance['importance_mean']:.4f}, std={sentiment_importance['importance_std']:.4f}")

print("\nTop 10 most important features (including sentiment):")
print(importance_df.head(10))

# Save importance report
importance_df.to_csv("feature_importance.csv", index=False)
with open("feature_importance_summary.txt", "w") as f:
    f.write(f"Sentiment Importance:\nMean: {sentiment_importance['importance_mean']:.4f}, Std: {sentiment_importance['importance_std']:.4f}\n\n")
    f.write("Top 10 Features:\n")
    f.write(importance_df.head(10).to_string(index=False))

print("Feature importance saved to 'feature_importance.csv' and 'feature_importance_summary.txt'")

