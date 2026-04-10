import pandas as pd

INPUT_FEATURE_FILE = r"D:\Final year project\code\feature_engineered_dataset.xlsx"

DATASETS = {
    "cresci17": {
        "emb": "cresci17_node_embeddings.csv",
        "label": "cresci17_labels.csv",
        "train_ids": "cresci17_train_ids.csv",
        "test_ids": "cresci17_test_ids.csv"
    },
    "twibot22": {
        "emb": "twibot22_node_embeddings.csv",
        "label": "twibot22_labels.csv",
        "train_ids": "twibot22_train_ids.csv",
        "test_ids": "twibot22_test_ids.csv"
    }
}

df = pd.read_excel(INPUT_FEATURE_FILE)
df["user_id"] = df["user_id"].astype(str)

all_train = []
all_test = []

for name, cfg in DATASETS.items():

    feat_df = df[df["dataset"] == name].copy()

    emb_df = pd.read_csv(cfg["emb"], dtype={"user_id": str})
    label_df = pd.read_csv(cfg["label"], dtype={"user_id": str})

    merged = feat_df.merge(emb_df, on="user_id")
    merged = merged.merge(label_df, on="user_id")

    train_ids = pd.read_csv(cfg["train_ids"], dtype={"user_id": str})["user_id"]
    test_ids = pd.read_csv(cfg["test_ids"], dtype={"user_id": str})["user_id"]

    train_df = merged[merged["user_id"].isin(train_ids)]
    test_df = merged[merged["user_id"].isin(test_ids)]

    all_train.append(train_df)
    all_test.append(test_df)

# ---------- combine both datasets ----------
final_train = pd.concat(all_train, ignore_index=True)
final_test = pd.concat(all_test, ignore_index=True)

final_train.to_csv("combined_train_merged.csv", index=False)
final_test.to_csv("combined_test_merged.csv", index=False)

print("Combined train/test files created.")
