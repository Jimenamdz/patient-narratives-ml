import os
import numpy as np
import pandas as pd
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg') 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Create output directory
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Setup logging
logging.basicConfig(filename=os.path.join(output_dir, 'model_training.log'),
                    level=logging.INFO,
                    format='%(asctime)s %(message)s')

logging.info("Script started.")

# =====================
# Step 1: Load Data
# =====================
terminal_embeddings = np.load("terminal_embeddings.npy")
non_terminal_embeddings = np.load("non_terminal_embeddings.npy")
terminal_sentiment = np.load("terminal_sentiment.npy")
non_terminal_sentiment = np.load("non_terminal_sentiment.npy")

terminal_df = pd.DataFrame(terminal_embeddings)
terminal_df['sentiment'] = terminal_sentiment
terminal_df['label'] = 1

non_terminal_df = pd.DataFrame(non_terminal_embeddings)
non_terminal_df['sentiment'] = non_terminal_sentiment
non_terminal_df['label'] = 0

full_df = pd.concat([terminal_df, non_terminal_df], ignore_index=True)
full_df.columns = full_df.columns.astype(str)
full_df.to_csv(os.path.join(output_dir, "features_and_sentiment.csv"), index=False)
logging.info("Combined features saved.")

# ==========================
# Step 2: Train-Test Split
# ==========================
X = full_df.drop('label', axis=1)
y = full_df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Balance data using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
logging.info("Data balanced using SMOTE.")

# ==========================
# Step 3: Model Pipelines
# ==========================

# Logistic Regression pipeline
logreg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

# Random Forest pipeline
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

# Logistic Regression hyperparameters
logreg_params = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l2']
}

# Random Forest hyperparameters
rf_params = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5, 10]
}

# ================================
# Step 4: Hyperparameter Tuning
# ================================

logreg_grid = GridSearchCV(logreg_pipeline, logreg_params, cv=5, scoring='roc_auc', n_jobs=-1)
logreg_grid.fit(X_train_balanced, y_train_balanced)
logging.info(f"LogReg best params: {logreg_grid.best_params_}")

rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=5, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X_train_balanced, y_train_balanced)
logging.info(f"Random Forest best params: {rf_grid.best_params_}")

# ==========================
# Step 5: Evaluate Models
# ==========================

def evaluate_model(grid, name):
    y_pred = grid.predict(X_test)
    y_pred_proba = grid.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=['Non-Terminal', 'Terminal'])
    auc = roc_auc_score(y_test, y_pred_proba)

    # Save classification report
    with open(os.path.join(output_dir, f"{name}_classification_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nROC-AUC: {auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Non-Terminal', 'Terminal']).plot(cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.savefig(os.path.join(output_dir, f"{name}_confusion_matrix.png"))
    plt.close()

    # ROC Curve
    RocCurveDisplay.from_estimator(grid, X_test, y_test)
    plt.title(f'{name} ROC Curve (AUC={auc:.2f})')
    plt.savefig(os.path.join(output_dir, f"{name}_roc_curve.png"))
    plt.close()

    logging.info(f"{name} evaluation done: ROC-AUC {auc:.4f}")

evaluate_model(logreg_grid, "Logistic_Regression")
evaluate_model(rf_grid, "Random_Forest")

# ==============================
# Step 6: Permutation Importance
# ==============================
perm_importance = permutation_importance(
    rf_grid, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1
)

# Build feature importance DataFrame from the model
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": perm_importance.importances_mean,
    "importance_std": perm_importance.importances_std
})

# Load the original feature sources to map back to Empath
terminal_combined = pd.read_csv("terminal_combined_features.csv")
non_terminal_combined = pd.read_csv("non_terminal_combined_features.csv")
combined_full = pd.concat([terminal_combined, non_terminal_combined], ignore_index=True)

# Identify which columns are embeddings and which are empath scores
embedding_feature_cols = [col for col in combined_full.columns if col.startswith("embedding_")]
empath_feature_cols = [col for col in combined_full.columns if col.startswith("empath_")]

# Compute correlation matrix to match embeddings to empath categories
correlations = combined_full[embedding_feature_cols + empath_feature_cols].corr()
correlations_subset = correlations.loc[embedding_feature_cols, empath_feature_cols]

# Build mapping from embedding feature to best empath category
correlation_mapping = correlations_subset.abs().idxmax(axis=1).to_dict()

# Map from numeric string (e.g., '522') to correlated empath label using the original embedding column
importance_df["corrected_empath_label"] = importance_df["feature"].apply(
    lambda x: correlation_mapping.get(f"embedding_{x}", "sentiment") if x != "sentiment" else "sentiment"
)

# Construct new label: "305 (empath_affection)"
importance_df["feature"] = importance_df.apply(
    lambda row: f"{row['feature']} ({row['corrected_empath_label']})" if row['corrected_empath_label'] != 'sentiment' else 'sentiment',
    axis=1
)

# Save and visualize
importance_df_sorted = importance_df.sort_values("importance_mean", ascending=False)
importance_df_sorted.to_csv(os.path.join(output_dir, "rf_feature_importance_with_empath.csv"), index=False)

plt.figure(figsize=(10,6))
sns.barplot(x='importance_mean', y='feature', data=importance_df_sorted.head(15), palette="coolwarm")
plt.title('Top 15 Random Forest Features with Empath Categories')
plt.xlabel('Permutation Importance Mean')
plt.ylabel('Feature (Empath Category)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rf_feature_importance_empath.png"))
plt.close()

logging.info("Final corrected feature importance analysis saved.")
print(f"All outputs saved in '{output_dir}' folder.")
