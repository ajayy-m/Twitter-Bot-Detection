import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer

# ==========================================
# FILE PATHS
# ==========================================
TRAIN_FILE = "Combined_Train_merged.csv"
TEST_FILE = "Combined_Test_merged.csv"

OUTPUT_TRAIN = "Normalized_train_dataset.csv"
OUTPUT_TEST = "Normalized_test_dataset.csv"

# ==========================================
# LOAD DATA (Ensure user_id is STRING)
# ==========================================
train_df = pd.read_csv(TRAIN_FILE, dtype={"user_id": str})
test_df = pd.read_csv(TEST_FILE, dtype={"user_id": str})

# ==========================================
# DEFINE COLUMN GROUPS
# ==========================================

EXCLUDE_COLUMNS = ["user_id", "label"]
account_age_col = "account_age_days"

# ---- Embedding columns ("0" to "63") ----
embedding_cols = [str(i) for i in range(64)]
embedding_cols = [col for col in embedding_cols if col in train_df.columns]

# ---- Detect Boolean columns (only 0/1) ----
boolean_cols = []
for col in train_df.columns:
    if col not in EXCLUDE_COLUMNS and col not in embedding_cols:
        unique_vals = train_df[col].dropna().unique()
        if len(unique_vals) > 0 and set(unique_vals).issubset({0, 1}):
            boolean_cols.append(col)

# ---- FIX: Restrict to numeric columns only ----
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

# ---- RobustScaler columns ----
robust_cols = [
    col for col in numeric_cols
    if col not in EXCLUDE_COLUMNS
    and col not in embedding_cols
    and col not in boolean_cols
    and col != account_age_col
]

print("Embedding Columns:", embedding_cols)
print("Boolean Columns:", boolean_cols)
print("RobustScaler Columns:", robust_cols)
print("StandardScaler Column:", account_age_col)

# ==========================================
# INITIALIZE SCALERS
# ==========================================
robust_scaler = RobustScaler()
standard_scaler = StandardScaler()
l2_normalizer = Normalizer(norm="l2")

# ==========================================
# FIT ON TRAIN, TRANSFORM BOTH
# ==========================================

# ---- RobustScaler ----
if robust_cols:
    train_df[robust_cols] = robust_scaler.fit_transform(train_df[robust_cols])
    test_df[robust_cols] = robust_scaler.transform(test_df[robust_cols])

# ---- StandardScaler (Account Age) ----
if account_age_col in train_df.columns:
    train_df[[account_age_col]] = standard_scaler.fit_transform(train_df[[account_age_col]])
    test_df[[account_age_col]] = standard_scaler.transform(test_df[[account_age_col]])

# ---- L2 Normalization (Embeddings) ----
if embedding_cols:
    train_df[embedding_cols] = l2_normalizer.fit_transform(train_df[embedding_cols])
    test_df[embedding_cols] = l2_normalizer.transform(test_df[embedding_cols])

# ==========================================
# ENSURE user_id remains STRING
# ==========================================
train_df["user_id"] = train_df["user_id"].astype(str)
test_df["user_id"] = test_df["user_id"].astype(str)

# ==========================================
# SAVE OUTPUT FILES
# ==========================================
train_df.to_csv(OUTPUT_TRAIN, index=False)
test_df.to_csv(OUTPUT_TEST, index=False)

import joblib

# ==========================================
# SAVE SCALERS FOR INFERENCE
# ==========================================
joblib.dump(robust_scaler, "robust_scaler.pkl")
joblib.dump(standard_scaler, "age_scaler.pkl")
joblib.dump(l2_normalizer, "embedding_normalizer.pkl")

print("Scalers saved successfully.")

print("\nNormalization Complete.")
print(f"Saved: {OUTPUT_TRAIN}")
print(f"Saved: {OUTPUT_TEST}")