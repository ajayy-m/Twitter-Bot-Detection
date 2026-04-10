import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from sklearn.feature_selection import mutual_info_classif

# ======================================
# FILES
# ======================================
TRAIN_FILE = "normalized_train_dataset.csv"
TEST_FILE  = "normalized_test_dataset.csv"

# ======================================
# LOAD DATA
# ======================================
train_df = pd.read_csv(TRAIN_FILE)
test_df  = pd.read_csv(TEST_FILE)

y_train = train_df["label"]
y_test  = test_df["label"]

# ======================================
# REMOVE TARGET COLUMNS
# ======================================
TARGET_COLS = ["label", "Label"]

train_df = train_df.drop(columns=[c for c in TARGET_COLS if c in train_df.columns])
test_df  = test_df.drop(columns=[c for c in TARGET_COLS if c in test_df.columns])

# ======================================
# FEATURE GROUPS
# ======================================
emb_cols = [c for c in train_df.columns if c.startswith("emb_") or c.isdigit()]
non_feature = ["user_id"] + emb_cols

ml_cols = [
    c for c in train_df.columns
    if c not in non_feature and
       pd.api.types.is_numeric_dtype(train_df[c])
]

# ======================================
# FEATURE SELECTION (MI + TREE IMPORTANCE)
# ======================================
X_ml = train_df[ml_cols].fillna(0)

mi_scores = mutual_info_classif(X_ml, y_train, random_state=42)
mi_series = pd.Series(mi_scores, index=ml_cols)

mi_keep = mi_series.sort_values(ascending=False)
mi_keep = mi_keep.iloc[:int(len(mi_keep)*0.7)].index.tolist()

print(f"MI selected {len(mi_keep)} ML features")

xgb_fs = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    n_estimators=300,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

xgb_fs.fit(train_df[mi_keep], y_train)

importance = pd.Series(xgb_fs.feature_importances_, index=mi_keep)

final_ml_cols = importance.sort_values(ascending=False)
final_ml_cols = final_ml_cols.iloc[:int(len(final_ml_cols)*0.8)].index.tolist()

print(f"Final ML features after tree selection: {len(final_ml_cols)}")

setups = {
    "ML only": final_ml_cols,
    "Embeddings only": emb_cols,
    "ML + Embeddings": final_ml_cols + emb_cols
}

# ======================================
# MODELS
# ======================================
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

models = {
    "XGBoost": XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        learning_rate=0.025,
        n_estimators=800,
        max_depth=5,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=1.5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=800,
        max_depth=18,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}

best_rf_score = -1
best_rf_model = None
best_rf_features = None
best_rf_setup = None

# ======================================
# TRAINING LOOP
# ======================================
results = []

for model_name, model in models.items():

    print(f"\n================ MODEL: {model_name} ================")

    for setup_name, cols in setups.items():

        print(f"Training setup: {setup_name}")

        X_train = train_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        X_test  = test_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        scores = {
            "Model": model_name,
            "Setup": setup_name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1": f1_score(y_test, preds),
            "AUC": roc_auc_score(y_test, y_prob)
        }

        results.append(scores)

        # ✅ SAVE BEST RANDOM FOREST INSIDE LOOP
        if model_name == "RandomForest":

            current_f1 = scores["F1"]

            if current_f1 > best_rf_score:
                best_rf_score = current_f1
                best_rf_model = model
                best_rf_features = cols
                best_rf_setup = setup_name

                print("New best RandomForest model found. Saving...")

                joblib.dump(best_rf_model, "random_forest_bot_classifier.pkl")
                joblib.dump(best_rf_features, "random_forest_feature_list.pkl")
                joblib.dump(final_ml_cols, "random_forest_ml_feature_cols.pkl")

# ======================================
# RESULTS
# ======================================
res_df = pd.DataFrame(results)[
    ["Model","Setup","Accuracy","Precision","Recall","F1","AUC"]
]

print("\nFinal Results")
print(res_df.to_string(index=False))

print("\nBest RandomForest Setup:", best_rf_setup)
print("Best RandomForest F1:", best_rf_score)